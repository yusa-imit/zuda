# zuda FFI Bindings

This directory contains examples of using zuda from other programming languages via FFI (Foreign Function Interface).

## Building the Shared Library

To use zuda from other languages, you must first build it as a shared library:

```bash
zig build -Dshared=true -Doptimize=ReleaseFast
```

This will produce:
- **macOS**: `zig-out/lib/libzuda.dylib`
- **Linux**: `zig-out/lib/libzuda.so`
- **Windows**: `zig-out/lib/zuda.dll`
- **Header**: `zig-out/include/zuda.h`

## Python Bindings (ctypes)

**Requirements**: Python 3.7+, no additional packages needed (uses built-in `ctypes`)

**Usage**:
```bash
python3 examples/python_bindings.py
```

**Features**:
- HashMap with integer keys/values
- BloomFilter with string membership testing
- Automatic memory management via Python destructors

**Example**:
```python
from python_bindings import HashMap, BloomFilter

# HashMap
hm = HashMap()
hm.put(1, 100)
print(hm.get(1))  # 100
hm.remove(1)

# BloomFilter
bf = BloomFilter(capacity=1000, false_positive_rate=0.01)
bf.insert("apple")
print(bf.contains("apple"))  # True
```

## Node.js Bindings (ffi-napi)

**Requirements**: Node.js 12+, ffi-napi, ref-napi

**Installation**:
```bash
cd examples
npm install ffi-napi ref-napi
```

**Usage**:
```bash
node examples/nodejs_bindings.js
```

**Features**:
- HashMap with BigInt keys/values (64-bit integers)
- BloomFilter with string membership testing
- Manual memory management (call `.destroy()` when done)

**Example**:
```javascript
const { HashMap, BloomFilter } = require('./nodejs_bindings');

// HashMap
const hm = new HashMap();
hm.put(1, 100);
console.log(hm.get(1));  // 100
hm.destroy();

// BloomFilter
const bf = new BloomFilter(1000, 0.01);
bf.insert('apple');
console.log(bf.contains('apple'));  // true
bf.destroy();
```

## C/C++ Usage

**Direct C usage**:
```c
#include <zuda.h>
#include <stdio.h>

// Define hash and comparison functions
uint64_t hash_int(const void* key, size_t size) {
    int64_t k = *(const int64_t*)key;
    return (uint64_t)(k < 0 ? -k : k);
}

int cmp_int(const void* a, const void* b, size_t size) {
    int64_t x = *(const int64_t*)a;
    int64_t y = *(const int64_t*)b;
    return (x == y) ? 0 : 1;
}

int main() {
    ZudaAllocator* alloc = zuda_allocator_create();
    ZudaHashMap* map = zuda_hashmap_create(alloc, 8, 8, hash_int, cmp_int);

    int64_t key = 42;
    int64_t value = 100;
    zuda_hashmap_put(map, &key, &value);

    int64_t result;
    if (zuda_hashmap_get(map, &key, &result) == 0) {
        printf("Value: %lld\n", result);
    }

    zuda_hashmap_destroy(map);
    zuda_allocator_destroy(alloc);
    return 0;
}
```

**Compile**:
```bash
gcc -o example example.c -L./zig-out/lib -lzuda -I./zig-out/include
LD_LIBRARY_PATH=./zig-out/lib ./example
```

## Available Data Structures

Currently, the C API exposes:

| Structure | Operations | Time Complexity |
|-----------|-----------|-----------------|
| **HashMap** | put, get, remove, count | O(1) average |
| **SkipList** | insert, get, remove, count | O(log n) |
| **BloomFilter** | insert, contains, count | O(k) where k = # hash functions |

More structures will be added in future releases.

## Memory Management

**Important**: All data structures require explicit memory management:

1. **Create allocator**: `zuda_allocator_create()`
2. **Create structure**: e.g., `zuda_hashmap_create(...)`
3. **Use structure**: insert, get, remove operations
4. **Destroy structure**: e.g., `zuda_hashmap_destroy(...)`
5. **Destroy allocator**: `zuda_allocator_destroy(...)`

For Python/Node.js wrappers, steps 4-5 are handled automatically (Python via `__del__`, Node.js via explicit `.destroy()` call).

## Performance Notes

- The C API uses opaque byte-level key/value storage for maximum flexibility
- For optimal performance in native Zig code, use the Zig API directly (comptime generics)
- The shared library is compiled with `-Doptimize=ReleaseFast` for production use

## Extending the Bindings

To add bindings for additional zuda structures:

1. Add C exports to `src/ffi/c_api.zig`
2. Update `include/zuda.h` with function declarations
3. Update language-specific bindings (Python, Node.js)
4. Add usage examples to this README

## Troubleshooting

**Library not found**:
- Ensure `zig build -Dshared=true` completed successfully
- Check that library path in bindings matches your OS (.so/.dylib/.dll)
- On Linux: set `LD_LIBRARY_PATH=./zig-out/lib`
- On macOS: set `DYLD_LIBRARY_PATH=./zig-out/lib`

**Segmentation fault**:
- Verify all pointers are valid before calling FFI functions
- Ensure proper memory alignment for key/value data
- Check that structures are not used after being destroyed

**Test the shared library**:
```bash
# macOS
otool -L zig-out/lib/libzuda.dylib

# Linux
ldd zig-out/lib/libzuda.so

# Check exported symbols
nm -D zig-out/lib/libzuda.so | grep zuda_
```
