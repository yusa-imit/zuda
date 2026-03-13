/**
 * zuda - Zig Universal Data Structures and Algorithms
 * C API Header
 *
 * This header provides C-compatible bindings for the zuda library.
 */

#ifndef ZUDA_H
#define ZUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * Memory Management
 * ========================================================================= */

/**
 * Opaque handle to a zuda allocator (wraps libc malloc/free)
 */
typedef struct ZudaAllocator ZudaAllocator;

/**
 * Create a C allocator using libc malloc/free
 * @return Allocator handle, or NULL on failure
 */
ZudaAllocator* zuda_allocator_create(void);

/**
 * Destroy an allocator
 * @param alloc Allocator handle
 */
void zuda_allocator_destroy(ZudaAllocator* alloc);

/* ============================================================================
 * HashMap
 * ========================================================================= */

/**
 * Opaque handle to a HashMap
 */
typedef struct ZudaHashMap ZudaHashMap;

/**
 * Hash function signature for HashMap keys
 * @param key Pointer to key data
 * @param key_size Size of key in bytes
 * @return Hash value
 */
typedef uint64_t (*ZudaHashFn)(const void* key, size_t key_size);

/**
 * Comparison function signature for HashMap keys
 * @param a First key
 * @param b Second key
 * @param size Size of keys in bytes
 * @return 0 if equal, non-zero otherwise
 */
typedef int (*ZudaCmpFn)(const void* a, const void* b, size_t size);

/**
 * Create a new HashMap
 * @param alloc Allocator handle
 * @param key_size Size of each key in bytes
 * @param value_size Size of each value in bytes
 * @param hash_fn Hash function for keys
 * @param cmp_fn Comparison function for keys
 * @return HashMap handle, or NULL on failure
 */
ZudaHashMap* zuda_hashmap_create(
    ZudaAllocator* alloc,
    size_t key_size,
    size_t value_size,
    ZudaHashFn hash_fn,
    ZudaCmpFn cmp_fn
);

/**
 * Destroy a HashMap and free all memory
 * @param map HashMap handle
 */
void zuda_hashmap_destroy(ZudaHashMap* map);

/**
 * Insert or update a key-value pair
 * @param map HashMap handle
 * @param key Pointer to key data
 * @param value Pointer to value data
 * @return 0 on success, -1 on error
 */
int zuda_hashmap_put(ZudaHashMap* map, const void* key, const void* value);

/**
 * Get a value by key
 * @param map HashMap handle
 * @param key Pointer to key data
 * @param out_value Pointer to buffer to write value (must be value_size bytes)
 * @return 0 if found, -1 if not found
 */
int zuda_hashmap_get(ZudaHashMap* map, const void* key, void* out_value);

/**
 * Remove a key-value pair
 * @param map HashMap handle
 * @param key Pointer to key data
 * @return 0 if removed, -1 if not found
 */
int zuda_hashmap_remove(ZudaHashMap* map, const void* key);

/**
 * Get the number of entries in the map
 * @param map HashMap handle
 * @return Number of entries
 */
size_t zuda_hashmap_count(ZudaHashMap* map);

/* ============================================================================
 * SkipList
 * ========================================================================= */

/**
 * Opaque handle to a SkipList
 */
typedef struct ZudaSkipList ZudaSkipList;

/**
 * Create a new SkipList
 * @param alloc Allocator handle
 * @param key_size Size of each key in bytes
 * @param value_size Size of each value in bytes
 * @param cmp_fn Comparison function for keys (returns <0, 0, >0)
 * @return SkipList handle, or NULL on failure
 */
ZudaSkipList* zuda_skiplist_create(
    ZudaAllocator* alloc,
    size_t key_size,
    size_t value_size,
    int (*cmp_fn)(const void*, const void*, size_t)
);

/**
 * Destroy a SkipList
 * @param list SkipList handle
 */
void zuda_skiplist_destroy(ZudaSkipList* list);

/**
 * Insert a key-value pair (maintains sorted order)
 * @param list SkipList handle
 * @param key Pointer to key data
 * @param value Pointer to value data
 * @return 0 on success, -1 on error
 */
int zuda_skiplist_insert(ZudaSkipList* list, const void* key, const void* value);

/**
 * Search for a key
 * @param list SkipList handle
 * @param key Pointer to key data
 * @param out_value Pointer to buffer to write value
 * @return 0 if found, -1 if not found
 */
int zuda_skiplist_get(ZudaSkipList* list, const void* key, void* out_value);

/**
 * Remove a key
 * @param list SkipList handle
 * @param key Pointer to key data
 * @return 0 if removed, -1 if not found
 */
int zuda_skiplist_remove(ZudaSkipList* list, const void* key);

/**
 * Get count of entries
 * @param list SkipList handle
 * @return Number of entries
 */
size_t zuda_skiplist_count(ZudaSkipList* list);

/* ============================================================================
 * BloomFilter
 * ========================================================================= */

/**
 * Opaque handle to a BloomFilter
 */
typedef struct ZudaBloomFilter ZudaBloomFilter;

/**
 * Create a BloomFilter
 * @param alloc Allocator handle
 * @param capacity Expected number of elements
 * @param false_positive_rate Desired false positive rate (e.g., 0.01 for 1%)
 * @return BloomFilter handle, or NULL on failure
 */
ZudaBloomFilter* zuda_bloomfilter_create(
    ZudaAllocator* alloc,
    size_t capacity,
    double false_positive_rate
);

/**
 * Destroy a BloomFilter
 * @param filter BloomFilter handle
 */
void zuda_bloomfilter_destroy(ZudaBloomFilter* filter);

/**
 * Insert an element
 * @param filter BloomFilter handle
 * @param data Pointer to element data
 * @param len Length of element in bytes
 */
void zuda_bloomfilter_insert(ZudaBloomFilter* filter, const void* data, size_t len);

/**
 * Check if element might be in the set
 * @param filter BloomFilter handle
 * @param data Pointer to element data
 * @param len Length of element in bytes
 * @return 1 if possibly present, 0 if definitely not present
 */
int zuda_bloomfilter_contains(ZudaBloomFilter* filter, const void* data, size_t len);

/**
 * Get the number of elements inserted
 * @param filter BloomFilter handle
 * @return Number of elements inserted
 */
size_t zuda_bloomfilter_count(ZudaBloomFilter* filter);

#ifdef __cplusplus
}
#endif

#endif /* ZUDA_H */
