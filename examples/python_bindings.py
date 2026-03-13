#!/usr/bin/env python3
"""
Python bindings for zuda library using ctypes

Example usage of zuda data structures from Python.
"""

import ctypes
import sys
from pathlib import Path

# Load the zuda shared library
# Adjust path as needed based on build output
lib_path = Path(__file__).parent.parent / "zig-out" / "lib" / "libzuda.so"
if not lib_path.exists():
    # Try .dylib on macOS
    lib_path = lib_path.with_suffix(".dylib")
if not lib_path.exists():
    # Try .dll on Windows
    lib_path = lib_path.with_suffix(".dll")

if not lib_path.exists():
    print(f"Error: Could not find zuda library at {lib_path}")
    print("Please build the library first: zig build -Dshared=true")
    sys.exit(1)

zuda = ctypes.CDLL(str(lib_path))

# ============================================================================
# Type Definitions
# ============================================================================

# Opaque pointer types
class ZudaAllocator(ctypes.Structure):
    pass

class ZudaHashMap(ctypes.Structure):
    pass

class ZudaSkipList(ctypes.Structure):
    pass

class ZudaBloomFilter(ctypes.Structure):
    pass

# Function pointer types
ZudaHashFn = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t)
ZudaCmpFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)

# ============================================================================
# Allocator Functions
# ============================================================================

zuda.zuda_allocator_create.argtypes = []
zuda.zuda_allocator_create.restype = ctypes.POINTER(ZudaAllocator)

zuda.zuda_allocator_destroy.argtypes = [ctypes.POINTER(ZudaAllocator)]
zuda.zuda_allocator_destroy.restype = None

# ============================================================================
# HashMap Functions
# ============================================================================

zuda.zuda_hashmap_create.argtypes = [
    ctypes.POINTER(ZudaAllocator),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ZudaHashFn,
    ZudaCmpFn,
]
zuda.zuda_hashmap_create.restype = ctypes.POINTER(ZudaHashMap)

zuda.zuda_hashmap_destroy.argtypes = [ctypes.POINTER(ZudaHashMap)]
zuda.zuda_hashmap_destroy.restype = None

zuda.zuda_hashmap_put.argtypes = [ctypes.POINTER(ZudaHashMap), ctypes.c_void_p, ctypes.c_void_p]
zuda.zuda_hashmap_put.restype = ctypes.c_int

zuda.zuda_hashmap_get.argtypes = [ctypes.POINTER(ZudaHashMap), ctypes.c_void_p, ctypes.c_void_p]
zuda.zuda_hashmap_get.restype = ctypes.c_int

zuda.zuda_hashmap_remove.argtypes = [ctypes.POINTER(ZudaHashMap), ctypes.c_void_p]
zuda.zuda_hashmap_remove.restype = ctypes.c_int

zuda.zuda_hashmap_count.argtypes = [ctypes.POINTER(ZudaHashMap)]
zuda.zuda_hashmap_count.restype = ctypes.c_size_t

# ============================================================================
# SkipList Functions
# ============================================================================

zuda.zuda_skiplist_create.argtypes = [
    ctypes.POINTER(ZudaAllocator),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
]
zuda.zuda_skiplist_create.restype = ctypes.POINTER(ZudaSkipList)

zuda.zuda_skiplist_destroy.argtypes = [ctypes.POINTER(ZudaSkipList)]
zuda.zuda_skiplist_destroy.restype = None

zuda.zuda_skiplist_insert.argtypes = [ctypes.POINTER(ZudaSkipList), ctypes.c_void_p, ctypes.c_void_p]
zuda.zuda_skiplist_insert.restype = ctypes.c_int

zuda.zuda_skiplist_get.argtypes = [ctypes.POINTER(ZudaSkipList), ctypes.c_void_p, ctypes.c_void_p]
zuda.zuda_skiplist_get.restype = ctypes.c_int

zuda.zuda_skiplist_remove.argtypes = [ctypes.POINTER(ZudaSkipList), ctypes.c_void_p]
zuda.zuda_skiplist_remove.restype = ctypes.c_int

zuda.zuda_skiplist_count.argtypes = [ctypes.POINTER(ZudaSkipList)]
zuda.zuda_skiplist_count.restype = ctypes.c_size_t

# ============================================================================
# BloomFilter Functions
# ============================================================================

zuda.zuda_bloomfilter_create.argtypes = [
    ctypes.POINTER(ZudaAllocator),
    ctypes.c_size_t,
    ctypes.c_double,
]
zuda.zuda_bloomfilter_create.restype = ctypes.POINTER(ZudaBloomFilter)

zuda.zuda_bloomfilter_destroy.argtypes = [ctypes.POINTER(ZudaBloomFilter)]
zuda.zuda_bloomfilter_destroy.restype = None

zuda.zuda_bloomfilter_insert.argtypes = [ctypes.POINTER(ZudaBloomFilter), ctypes.c_void_p, ctypes.c_size_t]
zuda.zuda_bloomfilter_insert.restype = None

zuda.zuda_bloomfilter_contains.argtypes = [ctypes.POINTER(ZudaBloomFilter), ctypes.c_void_p, ctypes.c_size_t]
zuda.zuda_bloomfilter_contains.restype = ctypes.c_int

zuda.zuda_bloomfilter_count.argtypes = [ctypes.POINTER(ZudaBloomFilter)]
zuda.zuda_bloomfilter_count.restype = ctypes.c_size_t

# ============================================================================
# Python Wrapper Classes
# ============================================================================

class HashMap:
    """Python wrapper for zuda HashMap"""

    def __init__(self):
        self.alloc = zuda.zuda_allocator_create()
        if not self.alloc:
            raise MemoryError("Failed to create allocator")

        # Hash function for integer keys
        @ZudaHashFn
        def hash_fn(key, size):
            key_int = ctypes.cast(key, ctypes.POINTER(ctypes.c_int64)).contents.value
            # Simple hash for demonstration
            return abs(hash(key_int)) & 0xFFFFFFFFFFFFFFFF

        # Comparison function for integer keys
        @ZudaCmpFn
        def cmp_fn(a, b, size):
            a_int = ctypes.cast(a, ctypes.POINTER(ctypes.c_int64)).contents.value
            b_int = ctypes.cast(b, ctypes.POINTER(ctypes.c_int64)).contents.value
            return 0 if a_int == b_int else 1

        # Keep references to prevent garbage collection
        self._hash_fn = hash_fn
        self._cmp_fn = cmp_fn

        self.map = zuda.zuda_hashmap_create(
            self.alloc,
            ctypes.sizeof(ctypes.c_int64),  # key size
            ctypes.sizeof(ctypes.c_int64),  # value size
            hash_fn,
            cmp_fn,
        )
        if not self.map:
            zuda.zuda_allocator_destroy(self.alloc)
            raise MemoryError("Failed to create HashMap")

    def __del__(self):
        if hasattr(self, 'map') and self.map:
            zuda.zuda_hashmap_destroy(self.map)
        if hasattr(self, 'alloc') and self.alloc:
            zuda.zuda_allocator_destroy(self.alloc)

    def put(self, key: int, value: int):
        """Insert or update a key-value pair"""
        key_c = ctypes.c_int64(key)
        value_c = ctypes.c_int64(value)
        result = zuda.zuda_hashmap_put(
            self.map,
            ctypes.byref(key_c),
            ctypes.byref(value_c)
        )
        if result != 0:
            raise RuntimeError("Failed to insert key-value pair")

    def get(self, key: int) -> int:
        """Get value by key"""
        key_c = ctypes.c_int64(key)
        value_c = ctypes.c_int64()
        result = zuda.zuda_hashmap_get(
            self.map,
            ctypes.byref(key_c),
            ctypes.byref(value_c)
        )
        if result != 0:
            raise KeyError(f"Key {key} not found")
        return value_c.value

    def remove(self, key: int):
        """Remove a key-value pair"""
        key_c = ctypes.c_int64(key)
        result = zuda.zuda_hashmap_remove(self.map, ctypes.byref(key_c))
        if result != 0:
            raise KeyError(f"Key {key} not found")

    def __len__(self):
        """Get number of entries"""
        return zuda.zuda_hashmap_count(self.map)


class BloomFilter:
    """Python wrapper for zuda BloomFilter"""

    def __init__(self, capacity: int, false_positive_rate: float = 0.01):
        self.alloc = zuda.zuda_allocator_create()
        if not self.alloc:
            raise MemoryError("Failed to create allocator")

        self.filter = zuda.zuda_bloomfilter_create(
            self.alloc,
            capacity,
            false_positive_rate
        )
        if not self.filter:
            zuda.zuda_allocator_destroy(self.alloc)
            raise MemoryError("Failed to create BloomFilter")

    def __del__(self):
        if hasattr(self, 'filter') and self.filter:
            zuda.zuda_bloomfilter_destroy(self.filter)
        if hasattr(self, 'alloc') and self.alloc:
            zuda.zuda_allocator_destroy(self.alloc)

    def insert(self, item: str):
        """Insert an item (string)"""
        data = item.encode('utf-8')
        zuda.zuda_bloomfilter_insert(self.filter, data, len(data))

    def contains(self, item: str) -> bool:
        """Check if item might be in the set"""
        data = item.encode('utf-8')
        result = zuda.zuda_bloomfilter_contains(self.filter, data, len(data))
        return result == 1

    def __len__(self):
        """Get number of elements inserted"""
        return zuda.zuda_bloomfilter_count(self.filter)


# ============================================================================
# Example Usage
# ============================================================================

def example_hashmap():
    print("=== HashMap Example ===")
    hm = HashMap()

    # Insert some key-value pairs
    hm.put(1, 100)
    hm.put(2, 200)
    hm.put(3, 300)
    print(f"Inserted 3 entries, count: {len(hm)}")

    # Get values
    print(f"hm[1] = {hm.get(1)}")
    print(f"hm[2] = {hm.get(2)}")
    print(f"hm[3] = {hm.get(3)}")

    # Update a value
    hm.put(2, 250)
    print(f"After update: hm[2] = {hm.get(2)}")

    # Remove a key
    hm.remove(1)
    print(f"After removing key 1, count: {len(hm)}")

    try:
        hm.get(1)
    except KeyError:
        print("Key 1 no longer exists (expected)")

    print()


def example_bloomfilter():
    print("=== BloomFilter Example ===")
    bf = BloomFilter(capacity=1000, false_positive_rate=0.01)

    # Insert some items
    words = ["apple", "banana", "cherry", "date", "elderberry"]
    for word in words:
        bf.insert(word)
    print(f"Inserted {len(bf)} items")

    # Check membership
    for word in words:
        result = bf.contains(word)
        print(f"'{word}' in filter: {result}")

    # Check items not inserted (should be False, but might have false positives)
    test_words = ["fig", "grape", "honeydew"]
    for word in test_words:
        result = bf.contains(word)
        print(f"'{word}' in filter: {result} (should be False)")

    print()


if __name__ == "__main__":
    try:
        example_hashmap()
        example_bloomfilter()
        print("All examples completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
