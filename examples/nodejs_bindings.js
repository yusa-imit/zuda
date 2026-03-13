/**
 * Node.js bindings for zuda library using ffi-napi
 *
 * Install dependencies first:
 *   npm install ffi-napi ref-napi
 *
 * Build the shared library:
 *   zig build -Dshared=true
 */

const ffi = require('ffi-napi');
const ref = require('ref-napi');
const path = require('path');

// Determine library path based on platform
const libDir = path.join(__dirname, '..', 'zig-out', 'lib');
let libPath;
if (process.platform === 'darwin') {
    libPath = path.join(libDir, 'libzuda.dylib');
} else if (process.platform === 'win32') {
    libPath = path.join(libDir, 'zuda.dll');
} else {
    libPath = path.join(libDir, 'libzuda.so');
}

// Opaque pointer types
const ZudaAllocator = ref.refType(ref.types.void);
const ZudaHashMap = ref.refType(ref.types.void);
const ZudaBloomFilter = ref.refType(ref.types.void);

// Function pointer types
const ZudaHashFn = ffi.Function('uint64', ['pointer', 'size_t']);
const ZudaCmpFn = ffi.Function('int', ['pointer', 'pointer', 'size_t']);

// Load the zuda library
const zuda = ffi.Library(libPath, {
    // Allocator
    'zuda_allocator_create': [ZudaAllocator, []],
    'zuda_allocator_destroy': ['void', [ZudaAllocator]],

    // HashMap
    'zuda_hashmap_create': [ZudaHashMap, [ZudaAllocator, 'size_t', 'size_t', ZudaHashFn, ZudaCmpFn]],
    'zuda_hashmap_destroy': ['void', [ZudaHashMap]],
    'zuda_hashmap_put': ['int', [ZudaHashMap, 'pointer', 'pointer']],
    'zuda_hashmap_get': ['int', [ZudaHashMap, 'pointer', 'pointer']],
    'zuda_hashmap_remove': ['int', [ZudaHashMap, 'pointer']],
    'zuda_hashmap_count': ['size_t', [ZudaHashMap]],

    // BloomFilter
    'zuda_bloomfilter_create': [ZudaBloomFilter, [ZudaAllocator, 'size_t', 'double']],
    'zuda_bloomfilter_destroy': ['void', [ZudaBloomFilter]],
    'zuda_bloomfilter_insert': ['void', [ZudaBloomFilter, 'pointer', 'size_t']],
    'zuda_bloomfilter_contains': ['int', [ZudaBloomFilter, 'pointer', 'size_t']],
    'zuda_bloomfilter_count': ['size_t', [ZudaBloomFilter]],
});

/**
 * HashMap wrapper for JavaScript
 */
class HashMap {
    constructor() {
        this.alloc = zuda.zuda_allocator_create();
        if (this.alloc.isNull()) {
            throw new Error('Failed to create allocator');
        }

        // Hash function for 64-bit integer keys
        this.hashFn = ffi.Callback('uint64', ['pointer', 'size_t'], (keyPtr, size) => {
            const keyBuf = ref.reinterpret(keyPtr, 8, 0);
            const key = keyBuf.readBigInt64LE(0);
            // Simple hash (for demonstration)
            const h = Number(key < 0n ? -key : key) & 0xFFFFFFFF;
            return BigInt(h);
        });

        // Comparison function for 64-bit integer keys
        this.cmpFn = ffi.Callback('int', ['pointer', 'pointer', 'size_t'], (aPtr, bPtr, size) => {
            const aBuf = ref.reinterpret(aPtr, 8, 0);
            const bBuf = ref.reinterpret(bPtr, 8, 0);
            const a = aBuf.readBigInt64LE(0);
            const b = bBuf.readBigInt64LE(0);
            return a === b ? 0 : 1;
        });

        this.map = zuda.zuda_hashmap_create(
            this.alloc,
            8, // key size (int64)
            8, // value size (int64)
            this.hashFn,
            this.cmpFn
        );

        if (this.map.isNull()) {
            zuda.zuda_allocator_destroy(this.alloc);
            throw new Error('Failed to create HashMap');
        }
    }

    destroy() {
        if (this.map && !this.map.isNull()) {
            zuda.zuda_hashmap_destroy(this.map);
            this.map = null;
        }
        if (this.alloc && !this.alloc.isNull()) {
            zuda.zuda_allocator_destroy(this.alloc);
            this.alloc = null;
        }
    }

    put(key, value) {
        const keyBuf = Buffer.alloc(8);
        const valueBuf = Buffer.alloc(8);
        keyBuf.writeBigInt64LE(BigInt(key));
        valueBuf.writeBigInt64LE(BigInt(value));

        const result = zuda.zuda_hashmap_put(this.map, keyBuf, valueBuf);
        if (result !== 0) {
            throw new Error('Failed to insert key-value pair');
        }
    }

    get(key) {
        const keyBuf = Buffer.alloc(8);
        const valueBuf = Buffer.alloc(8);
        keyBuf.writeBigInt64LE(BigInt(key));

        const result = zuda.zuda_hashmap_get(this.map, keyBuf, valueBuf);
        if (result !== 0) {
            throw new Error(`Key ${key} not found`);
        }

        return Number(valueBuf.readBigInt64LE(0));
    }

    remove(key) {
        const keyBuf = Buffer.alloc(8);
        keyBuf.writeBigInt64LE(BigInt(key));

        const result = zuda.zuda_hashmap_remove(this.map, keyBuf);
        if (result !== 0) {
            throw new Error(`Key ${key} not found`);
        }
    }

    get size() {
        return zuda.zuda_hashmap_count(this.map);
    }
}

/**
 * BloomFilter wrapper for JavaScript
 */
class BloomFilter {
    constructor(capacity, falsePositiveRate = 0.01) {
        this.alloc = zuda.zuda_allocator_create();
        if (this.alloc.isNull()) {
            throw new Error('Failed to create allocator');
        }

        this.filter = zuda.zuda_bloomfilter_create(this.alloc, capacity, falsePositiveRate);
        if (this.filter.isNull()) {
            zuda.zuda_allocator_destroy(this.alloc);
            throw new Error('Failed to create BloomFilter');
        }
    }

    destroy() {
        if (this.filter && !this.filter.isNull()) {
            zuda.zuda_bloomfilter_destroy(this.filter);
            this.filter = null;
        }
        if (this.alloc && !this.alloc.isNull()) {
            zuda.zuda_allocator_destroy(this.alloc);
            this.alloc = null;
        }
    }

    insert(item) {
        const buf = Buffer.from(item, 'utf-8');
        zuda.zuda_bloomfilter_insert(this.filter, buf, buf.length);
    }

    contains(item) {
        const buf = Buffer.from(item, 'utf-8');
        const result = zuda.zuda_bloomfilter_contains(this.filter, buf, buf.length);
        return result === 1;
    }

    get size() {
        return zuda.zuda_bloomfilter_count(this.filter);
    }
}

// ============================================================================
// Example Usage
// ============================================================================

function exampleHashMap() {
    console.log('=== HashMap Example ===');
    const hm = new HashMap();

    try {
        // Insert some key-value pairs
        hm.put(1, 100);
        hm.put(2, 200);
        hm.put(3, 300);
        console.log(`Inserted 3 entries, count: ${hm.size}`);

        // Get values
        console.log(`hm[1] = ${hm.get(1)}`);
        console.log(`hm[2] = ${hm.get(2)}`);
        console.log(`hm[3] = ${hm.get(3)}`);

        // Update a value
        hm.put(2, 250);
        console.log(`After update: hm[2] = ${hm.get(2)}`);

        // Remove a key
        hm.remove(1);
        console.log(`After removing key 1, count: ${hm.size}`);

        try {
            hm.get(1);
        } catch (e) {
            console.log('Key 1 no longer exists (expected)');
        }
    } finally {
        hm.destroy();
    }

    console.log();
}

function exampleBloomFilter() {
    console.log('=== BloomFilter Example ===');
    const bf = new BloomFilter(1000, 0.01);

    try {
        // Insert some items
        const words = ['apple', 'banana', 'cherry', 'date', 'elderberry'];
        words.forEach(word => bf.insert(word));
        console.log(`Inserted ${bf.size} items`);

        // Check membership
        words.forEach(word => {
            const result = bf.contains(word);
            console.log(`'${word}' in filter: ${result}`);
        });

        // Check items not inserted
        const testWords = ['fig', 'grape', 'honeydew'];
        testWords.forEach(word => {
            const result = bf.contains(word);
            console.log(`'${word}' in filter: ${result} (should be false)`);
        });
    } finally {
        bf.destroy();
    }

    console.log();
}

// Run examples
if (require.main === module) {
    try {
        exampleHashMap();
        exampleBloomFilter();
        console.log('All examples completed successfully!');
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

module.exports = { HashMap, BloomFilter };
