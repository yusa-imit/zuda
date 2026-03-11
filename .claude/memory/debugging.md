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

## Common Data Structure Pitfalls
- Red-black tree: remember to handle both left and right uncle cases in fixup
- Skip list: randomized level generation must be bounded by max level
- Fibonacci heap: consolidate after extract-min, update min pointer
- B-Tree: split must propagate upward; handle root split as special case
- Hash table: rehash threshold must account for tombstones in open addressing

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
