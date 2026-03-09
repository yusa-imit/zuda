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
