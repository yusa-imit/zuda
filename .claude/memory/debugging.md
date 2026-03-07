# zuda Debugging Notes

## Fixed Issues
(none yet)

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
