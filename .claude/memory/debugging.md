# zuda Debugging Notes

## Fixed Issues

### PolyaAeppli quantile()/entropy() O(MAX_K²) hang from f32-underflowing epsilon literal (session 767, fixed in 524ead7)
**Symptoms**: `zig build test` hung indefinitely (observed 5+ min at 99.5% CPU on the test binary, no output). Bisected with `zig test src/stats/distributions.zig --test-filter "PolyaAeppli"` down to the "f32 type support" test and, separately, the "quantile(1) is defined" test.

**Root Cause**: Two independent bugs in the same pattern — a truncated-series-convergence check meant to break out of a `while (k <= MAX_K)` loop early once `pmf(k)` becomes negligible:
1. `if (p < 1e-300) break;` (in `entropy()`) and `if (k > 0 and pk < 1e-300) return k;` (in `quantile()`) used a comptime-float literal that gets coerced to `T` at the comparison site. For `T = f32`, `1e-300` underflows to exactly `0.0` (f32's minimum subnormal is ~1.4e-45), so the check silently became `p < 0.0` — never true for a non-negative PMF. The loop then ran the full `MAX_K = 50000` iterations, each O(k) (via `logpmf`'s inner sum), for O(MAX_K²) ≈ 2.5 billion ops.
2. Independently, even for `T = f64`, `quantile(1.0)` hung: the truncated PMF sum asymptotically plateaus around `0.99999294`, short of `1.0` by ~7e-6 — much larger than float epsilon, caused by accumulated approximation error in the Lanczos `logGamma` / Stirling `logFactorial` helpers used by `logpmf`. A first-attempt fix using a fixed `1e-9` absolute tolerance (`cumsum >= p - 1e-9`) was insufficient because the real gap exceeded it.

**Fix**: for both checks, compare against exact `0.0` instead of a magnitude literal:
```zig
// entropy(): was `if (p < 1e-300) break;`
if (p == 0.0) break;

// quantile(): was `if (k > 0 and pk < 1e-300) return k;`
if (k > 0 and pk == 0.0) return k;
```
Every strictly-decaying float series eventually underflows to exact zero at *some* k regardless of T's dynamic range — so `== 0.0` reliably terminates the loop without needing a per-type magic constant. `mode()` in the same file was already safe because its tolerance (`best_pmf * 1e-12`) is *relative* to the current best PMF, not absolute, so it scales automatically with T.

**Generalizable lesson**: never use a hardcoded absolute epsilon literal (`1e-300`, `1e-15`, ...) to detect "this float value has become negligible" when the code is generic over float type T — for narrower types (f32) the literal can underflow to `0.0` at compile time and silently disable the check. Use `== 0.0` for "has this series converged to nothing" or a tolerance relative to another value of the same type for "are these two values of the same type close". Two other distributions in `distributions.zig` (~line 71325, ~81533 as of this session) have the identical `if (p < 1e-300) break;` pattern; not fixed this session (already passing in CI, so no live bug — but only because their f32 test coverage doesn't currently exercise the same path). Flagged for a future STABILIZATION audit.

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

## Code Quality Issues

### Allocator-First Violations (RESOLVED - Session 554)
**Issue**: Hardcoded `std.heap.page_allocator` in library code (violates allocator-first principle)

**Impact**: Prevents memory leak detection, custom allocation strategies, embedded use

**Status**: ✅ **ALL FIXED** (0 violations remaining as of Session 554)

**Verification**: `grep -r "std\.heap\.page_allocator" src | grep -v "///"` returns 0 instances

**Remaining acceptable uses** (2 instances):
- algorithms/combinatorics/permutations.zig:44 — documentation example (`//!` comment)
- ffi/c_api.zig:16 — FFI layer using `std.heap.c_allocator` (correct for C interop)

**Fixed**:
- Session 440: tsp.zig (isValidTour), ridge_regression.zig (gaussianElimination)
- Session 417-419: LCS, LIS, catalan_numbers, perfect_squares, unique_paths
- Sessions 441-553: All remaining 27 violations resolved

**Solution Pattern** (for reference):
1. Add `allocator: Allocator` parameter to function signature
2. Thread through all call sites (tests and library code)
3. Update all test calls to use `std.testing.allocator`

**Lesson**: All library code now properly accepts allocators as parameters, enabling memory leak detection, custom allocation strategies, and embedded use cases.

## @panic in Library Code (Session 570, 2026-05-24)

**Issue**: 4 library functions used `@panic` instead of returning errors, violating the "No @panic" coding standard.

**Locations**:
1. `src/algorithms/sorting/bitonicsort.zig:60` — panic on non-power-of-2 length
2. `src/algorithms/sorting/bogosort.zig:41,141` — panic on getrandom failure
3. `src/algorithms/bitwise/subsets.zig:30,71,72` — panic on k>n or n>63 (n>63 was dead code since param is u6)
4. `src/stats/random.zig:262` — panic on lambda <= 0

**Fix Pattern**: Change return type from T/void to `error{ErrorName}!T/void`, replace `@panic` with `return error.ErrorName`.
- Callers inside the same file: add `try`
- Tests: update to use `try` or `expectError` for error cases

**Dead Code Discovery**: `subsets.zig` checked `n > 63` but parameter type is `u6` (max value = 63), making the condition always false. Removed the unreachable check.

**YuleSimon Variance Formula (2026-06-15)**: The Wikipedia article on Yule-Simon sometimes lists Var = ρ²(ρ+1)/((ρ-1)²(ρ-2)) but the correct formula derived by telescoping sums is Var = ρ²/((ρ-1)²(ρ-2)). For ρ=3: correct value is 2.25, not 9.0. Verified via partial fraction decomposition of E[X²] = Σ 18k/((k+3)(k+2)(k+1)) where the partial fractions telescope to a finite constant.

**YuleSimon Entropy ρ=1 (2026-06-15)**: Entropy ≈ 2·Σ log(k)/(k²-1) ≈ 2.026 nats. The test-writer guessed 2.5 which is wrong. Computed via integral approximation: 2·∫_1^∞ log(x)/x² dx = 2 with correction for lower terms.

## f32-Underflow Epsilon Bugs, Full Audit (Session 770, 2026-07-13)

**Issue**: `1e-300` as a literal compared/used at `T = f32` underflows to exact `0.0` at compile time (f32's smallest subnormal is ~1.4e-45). This silently breaks several distinct idioms in `distributions.zig`, not just loop-break checks (see the PolyaAeppli O(MAX_K²) hang from session 767).

**Full audit result** (18 `1e-300` sites checked):
- **`x < 1e-300` break checks** in unbounded-but-capped loops (`MAX_K = 50000`): becomes `x < 0.0`, never true. If `x` is a probability that itself underflows to exact `0.0` in f32, and the loop body does `sum -= x * @log(x)` or similar, this produces `0 * (-inf) = NaN`. **Fixed**: `Borel.entropy()`, `GeneralizedPoisson.entropy()` — now `if (p == 0.0) break`.
- **`x < 1e-300` division guards**: e.g. `ExponentiatedWeibull.hMode()` had `if (g < 1e-300) return <limit formula>` before dividing by `g`; vacuous for f32 means it divides by a value that may itself be exact `0.0`. **Fixed**: now `if (g == 0.0) return ...`.
- **`x == 0.0 ? 1e-300 : x` zero-replacement idiom** before `@log(x)`: the replacement value itself underflows to `0.0` in f32, so `@log` still sees `0` → `-inf`. Found in Box-Muller sampling: `LogitNormal.sample()`, `ExponentialModifiedGaussian.sample()`. **Fixed**: use `std.math.floatMin(T)` instead (the pattern already used correctly elsewhere in the file, e.g. `~line 14816`).
- **`@max(x, 1e-300)` clamp** before `@log(x)`: same issue, found in `gigLogBesselK` (GIG Bessel-K helper used by GeneralizedInverseGaussian/NormalInverseGaussian). **Fixed**: `@max(x, std.math.floatMin(T))`.
- **Safe, left unchanged**: guards written as `x > 1e-300` (underflowing to `x > 0.0` still correctly excludes the one bad case `x == 0`), and break checks inside loops with a small fixed bound (≤ ~500 iterations) where a vacuous check just means the loop runs to its bound instead of exiting early (no hang, no NaN).

**Verification method**: confirmed `@as(f32, 1e-300) == 0.0` and `@log(@as(f32,0.0)) == -inf` empirically via a throwaway `zig run` snippet, then compiled a small harness importing `zuda` as a module (`zig run --dep zuda -Mroot=... -Mzuda=src/root.zig`) exercising all 4 fixed paths at `T = f32` (200k Box-Muller samples each) — 0 NaN/Inf after the fix. Don't just reason about whether an f32 path is broken; compile and run it.

Commit: 5370a48.
