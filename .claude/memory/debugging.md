# zuda Debugging Notes

## Fixed Issues

### SuffixTree Edge Splitting Bug (Issue #1, fixed in d17ca50)
**Symptoms**: `findAll()` returned duplicate suffix indices (e.g., 3 instead of 2 for "ana" in "banana"); `longestRepeatedSubstring()` returned null for strings with repeated substrings.

**Root Cause**: Two separate bugs:
1. **Pattern search at edge boundaries**: When pattern match exhausted exactly at `j == edge.label.len`, the code didn't move `node` to `edge.target`. This left `node` pointing to a parent node, causing `collectLeaves()` to collect incorrect suffix indices from parent's subtree.
2. **LRS detection logic**: Only checked `node.children.count() >= 2`, but missed nodes with `suffix_index != null` AND `children.count() >= 1`. Such nodes represent a repeated substring (one occurrence ends at this node, another continues in a child path).

**Fix**:
```zig
// Before: ambiguous condition chain
if (i < pattern.len) {
    node = edge.target;
} else if (j < edge.label.len) {
    found_edge = edge;
}

// After: explicit exhausted case handling
if (i >= pattern.len) {
    if (j < edge.label.len) {
        found_edge = edge;
    } else {
        node = edge.target;  // MUST move to target when pattern ends at boundary
    }
} else {
    node = edge.target;
}

// LRS fix: recognize both branching patterns
const is_repeated = (node.children.count() >= 2) or
                   (node.suffix_index != null and node.children.count() >= 1);
```

**Lesson**: In compressed suffix trees, internal nodes can have `suffix_index` set when a suffix ends exactly at a branch point. Pattern search must explicitly handle all three cases: pattern continues, ends mid-edge, ends at boundary.

## Known Zig 0.15.x Gotchas (from sibling projects)
- `std.ArrayList(T){}` not `.init(allocator)` — unmanaged API
- `std.Thread.sleep(ns)` not `std.time.sleep`
- `child.wait()` closes stdout — read stdout BEFORE wait()
- `callconv(.c)` lowercase in 0.15
- Buffered writers: flush before `std.process.exit()`
- File-scope: `const X = expr;` (no `comptime` keyword — redundant error)
- `zig build test` uses `--listen=-` protocol — NEVER use `stdout()` in test code
- **std.atomic.fence() removed** (Issue #7, fixed in 44bf1f6):
  - Replace with stronger memory ordering on atomic ops (.seq_cst)
  - For lock-free data structures, upgrade .acquire/.release to .seq_cst where fence was used
  - Alternative: use dummy atomic RMW with .seq_cst as portable fence
- **Generic functions can't be comptime params** (Issue #8, fixed in 44bf1f6):
  - Problem: `fn hash(ctx: Context, key: anytype)` can't be passed as `comptime hashFn: fn(Context, K) u64`
  - Solution: Create concrete wrapper inside factory function with known K type
  - Pattern: Move AutoContext struct INSIDE Auto* factory, not as top-level export
- **128-bit atomics NOT universally supported** (fixed in e67fe1b):
  - `std.atomic.Value(u128)` requires CMPXCHG16B (x86-64) or CASP (ARM64)
  - **NOT supported**: Windows (max 64-bit), WASM (max 32-bit), Linux (not guaranteed)
  - **Supported**: macOS x86-64/ARM64 (Apple enforces CPU requirements)
  - **Error symptom**: `expected 64-bit integer type or smaller; found 128-bit integer type`
  - **Fix**: Use comptime check to restrict to macOS-only, OR rewrite using two separate atomics
  - **Affected**: LockFreeStack, LockFreeQueue (now macOS-only)
  - **Alternative**: WorkStealingDeque (portable, uses usize atomics)

## Common Data Structure Pitfalls
- Red-black tree: remember to handle both left and right uncle cases in fixup
- Skip list: randomized level generation must be bounded by max level
- Fibonacci heap: consolidate after extract-min, update min pointer
- B-Tree: split must propagate upward; handle root split as special case
- Hash table: rehash threshold must account for tombstones in open addressing

### FibonacciHeap Node Initialization Bug (fixed in 6485859)
**Symptoms**: Segfault during deinit, "Invalid free" panics, crashes even with 5 nodes. Previous investigation focused on O(n²) complexity was misleading.

**Root Cause**: Node.init() returns a stack-allocated struct with self-referential pointers (`node.prev = &node`). After copying this value to heap via `node.* = Node.init(value)`, the prev/next pointers still pointed to the stack copy, creating dangling pointers.

**Fix**:
```zig
// In insert():
const node = try self.allocator.create(Node);
node.* = Node.init(value);
// CRITICAL: Fix up circular pointers to point to allocated node
node.prev = node;
node.next = node;
```

**Lesson**: When a struct initializer returns a value with self-referential pointers, those pointers MUST be updated after copying to the heap. The stack address becomes invalid immediately. This is a subtle bug because it works initially but corrupts memory on traversal.

### Push-Relabel Infinite Loop (fixed in 02a920b)
**Symptoms**: Tests hang indefinitely on graphs with no path from source to sink.

**Root Cause**: Without a height bound, vertices with excess that can't reach the sink will have their heights relabeled indefinitely. The discharge function kept relabeling without termination.

**Fix**:
```zig
// Add height bound check in discharge()
const max_height = 2 * vertex_data.count(); // 2V is theoretical upper bound

while (true) {
    // ...
    // If height exceeds bound, vertex can't reach sink - stop processing
    if (u_data.height >= max_height) return;
    // ...
}
```

**Lesson**: Push-Relabel requires a height bound to prevent infinite relabeling. The standard bound is 2V (2 * number of vertices). Vertices that exceed this bound cannot reach the sink and should be skipped. Also, using a current-edge pointer in discharge() is crucial for efficiency and correctness.

### CI Timeout Due to Excessive Test Compilation (Issue #3, fixed in fd8a3cf)
**Symptoms**: `zig build test` hangs for 30+ minutes, CI times out, builds never complete.

**Root Cause**: Two compounding issues:
1. **main.zig importing zuda**: The demo executable did `@import("zuda")`, forcing semantic analysis of all 195 imports when compiling the executable. This was unnecessary since main.zig didn't actually use any zuda types.
2. **Massive test reftest block**: root.zig had a test block with 80+ manual `_ = Container/Algorithm` references. This forced compilation of hundreds of individual test suites into a single compilation unit, overwhelming the compiler.

**Fix**:
```zig
// main.zig - remove unused import
- const zuda = @import("zuda");
+ // NOTE: Import removed to avoid triggering semantic analysis during executable build

// root.zig - simplify test block
- test {
-     std.testing.refAllDecls(@This());
-     _ = containers.trees.BTree;
-     _ = containers.trees.RedBlackTree;
-     ... (80+ lines of manual references)
- }
+ test {
+     // refAllDecls is sufficient; individual module tests run via `zig build test`
+     std.testing.refAllDecls(@This());
+ }
```

**Result**: Build time reduced from >30min (timeout) to <3min.

**Lesson**:
- For large Zig libraries with 100+ modules, avoid importing the entire library in binaries that don't use it.
- Don't manually reference all types in a single test block - `refAllDecls` is sufficient.
- When `zig build test` hangs, check for: excessive test references, unused imports in executables, or circular test dependencies.
- Individual module tests are already discovered by `zig build test` - no need to manually aggregate them.

## Performance Issues

### RedBlackTree Performance Below Target (identified 2026-03-14, commit 232f2ad)
**Symptoms**: Benchmark shows insert at 269 ns/op (target ≤ 200 ns/op, 34.5% over) and lookup at 552 ns/op (target ≤ 150 ns/op, 268% over) for 1M random keys.

**Status**: Under investigation

**Potential Causes**:
1. Lookup performance particularly concerning (268% over) - suggests algorithmic issue or cache inefficiency
2. Random key distribution may cause poor cache locality during tree traversal
3. Possible excessive allocator overhead in node allocation
4. Tree rotations during insert may be more frequent than expected
5. Context comparison function overhead (though simple integer comparison should be fast)

**Next Steps**:
- Profile the hot path with perf/Instruments
- Compare against C++ std::map with same workload
- Test with sequential keys vs random to measure cache impact
- Consider node pooling instead of individual allocations
- Check if pointer chasing patterns can be optimized
- Verify tree height stays within expected O(log n) bounds
