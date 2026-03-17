# Double-Array Trie Cache Behavior Analysis

## Executive Summary

Current DoubleArrayTrie implementation: **125.0 MB/sec** (baseline)
Target performance: **≥180 MB/sec** (+44% improvement)

**Root cause**: Memory layout fragmentation across 4-5 separate arrays causes multiple cache misses per character processed.

**Solution**: Linearize state data into single contiguous buffer with struct-of-arrays (SoA) layout optimized for sequential access.

---

## Current Implementation Analysis

### Memory Layout

```
BASE:   [base₀, base₁, ..., baseₙ]    (4 bytes × n states)
CHECK:  [check₀, check₁, ..., checkₙ]  (4 bytes × n states)
FAIL:   [fail₀, fail₁, ..., failₙ]     (4 bytes × n states)
IS_LEAF:[leaf₀, leaf₁, ..., leafₙ]     (1 byte × n states)
OUTPUT: [out₀*, out₁*, ..., outₙ*]     (ArrayList × n states, ~24 bytes each)
```

**Total per state**: 4 + 4 + 4 + 1 + 24 = **37 bytes** (but spread across 5 allocations)

### Hot Path — `findAll()` Search Loop

```zig
// Line 413-419: Transition check (2 array accesses)
const base_val = self.base[current_state];          // ❶ Cache miss (random access)
const next_state = base_val + char;
if (self.check[next_state] == current_state) {     // ❷ Cache miss (dependent load)
    current_state = next_state;
    break;
}

// Line 433: Failure link (1 array access)
current_state = self.fail[current_state];           // ❸ Cache miss (random access)

// Line 441: Output (1 ArrayList indirection)
for (self.output[current_state].items) |pattern_idx| {  // ❹ Pointer chase (2 loads)
    ...
}
```

**Cache line analysis** (assuming 64-byte cache lines):

| Access | Array | Bytes | Cache lines loaded | Why |
|--------|-------|-------|-------------------|-----|
| ❶ | BASE | 4 | 1 | Load 64-byte cache line containing `base[current_state]` |
| ❷ | CHECK | 4 | 1 | `next_state` is random → different cache line |
| ❸ | FAIL | 4 | 1 (when taken) | Failure link = random access |
| ❹ | OUTPUT | 8 + deref | 2 (pointer + data) | ArrayList header + heap-allocated slice |

**Worst case**: 5 cache misses per character (when failure link is followed)
**Best case**: 2 cache misses per character (valid transition, no output)

**Estimated cache miss latency**: ~200 ns/miss (from v1.6.0 RedBlackTree analysis)

### Performance Bottleneck Calculation

**Current throughput**: 125.0 MB/sec
**Characters processed**: 125M chars/sec
**Cycles per character** (@ 3 GHz): 24 cycles/char

**Breakdown**:
- Cache misses: 2-5 × 200 ns = 400-1000 ns (dominates)
- Computation: ~10 ns (arithmetic, bounds checks)
- Branch mispredictions: ~5 ns

**Conclusion**: Memory-bound, not compute-bound.

---

## Proposed Optimization — Linearized Layout

### Strategy 1: Structure-of-Arrays (SoA) with Sequential Packing

Pack all state data into a single contiguous buffer:

```zig
const State = struct {
    base: i32,      // 4 bytes
    check: u32,     // 4 bytes
    fail: u32,      // 4 bytes
    output_start: u32,  // 4 bytes (index into separate output array)
    output_count: u16,  // 2 bytes
    is_leaf: bool,     // 1 byte
    _padding: u8,      // 1 byte (align to 20 bytes)
};
states: []State;  // Single allocation, 20 bytes per state
```

**Benefits**:
1. **Spatial locality**: All data for state `s` is within 20 bytes (fits in 1 cache line)
2. **Prefetching**: CPU prefetcher can load next state's data while processing current
3. **Reduced TLB pressure**: 1 allocation instead of 5

**Estimated improvement**:
- Cache misses: 2-5 → 1-2 per character (-50% to -60%)
- Throughput: 125.0 MB/sec → **200-250 MB/sec** (+60% to +100%)

### Strategy 2: Interleaved BASE+CHECK

For extremely tight loops, interleave only BASE and CHECK (most frequently accessed):

```zig
base_check: []BaseCheck,  // struct { base: i32, check: u32 }
fail: []u32,
output: []std.ArrayList(usize),
```

**Benefits**:
- BASE and CHECK guaranteed in same cache line (8 bytes total)
- Minimal code changes (only transition validation logic)

**Estimated improvement**:
- Cache misses: 2 → 1 for transition check (-50%)
- Throughput: 125.0 MB/sec → **160-180 MB/sec** (+28% to +44%)

---

## Implementation Plan

### Phase 1: Baseline Validation ✅

- [x] Create `bench/cache_profile_strings.zig` to measure current throughput
- [x] Establish baseline: 125.0 MB/sec (1000 patterns, 1 MB text)

### Phase 2: Interleaved BASE+CHECK (Quick Win) ✅ COMPLETE

**Goal**: Achieve ≥160 MB/sec (+28% improvement)

**Implementation** (2026-03-18):
1. ✅ Created `BaseCheck` struct (8 bytes: i32 base + u32 check)
2. ✅ Replaced separate `base: []i32, check: []u32` with `base_check: []BaseCheck`
3. ✅ Updated `init()`, `buildFailureLinks()`, `contains()`, `findAll()`, `validate()`
4. ✅ All 722 tests passing (correctness preserved)

**Results** (5 benchmark runs, 1000 patterns, 1 MB text):
- **Throughput**: 122.4 MB/sec (avg: 121.4, 121.9, 122.9, 122.9, 123.3)
- **vs Baseline**: -2.6 MB/sec (-2% regression)
- **vs Target**: -37.6 MB/sec (-24% below 160 MB/sec target)

**Root Cause Analysis**:
- ✅ **BaseCheck interleaving works**: Struct is 8 bytes, fits in cache line
- ❌ **But hot path still has 3-4 cache misses**:
  1. BASE+CHECK: 1 load (interleaved) ✅ — fixed
  2. FAIL[current]: separate array access ❌ — still fragmented
  3. OUTPUT[current]: ArrayList pointer dereference ❌ — still fragmented
  4. OUTPUT data: separate heap allocation ❌ — still fragmented
- **Bottleneck**: Interleaving BASE+CHECK fixed only **1 of 3-4** cache misses
- **Conclusion**: Phase 2 alone is **insufficient** for +28% target

**Code changes** (actual):
- `double_array_trie.zig`: ~80 lines modified (init, buildFailureLinks, deinit, contains, findAll, validate)
- `bench/cache_profile_strings.zig`: 5 lines modified (memory calculation)

**Risk**: LOW (minimal logic changes, all tests pass)

### Phase 3: Full Linearization (Stretch Goal)

**Goal**: Achieve ≥200 MB/sec (+60% improvement)

1. **Define `State` struct** (20 bytes):
   ```zig
   const State = packed struct {
       base: i32,
       check: u32,
       fail: u32,
       output_start: u32,
       output_count: u16,
       is_leaf: bool,
       _pad: u8,
   };
   ```
2. **Flatten OUTPUT**:
   - Replace `output: []std.ArrayList(usize)` with:
     - `states: []State` (includes `output_start`, `output_count`)
     - `output_data: []usize` (flat array of pattern indices)
   - During init: pack all OUTPUT lists sequentially into `output_data`
3. **Update all accessors**:
   - `contains()`: `states[s].base`, `states[s].check`
   - `findAll()`: `states[s].output_start .. output_start + output_count`
4. **Benchmark**: Measure throughput improvement
5. **Validate**: Run all 40 tests + stress tests (10k patterns)

**Code changes** (estimated):
- `double_array_trie.zig`: ~150 lines modified
- `aho_corasick.zig`: 0 lines (API unchanged)

**Risk**: MEDIUM (complex refactoring, must preserve all invariants)

### Phase 4: Prefetching Hints (Optional)

If still short of 200 MB/sec, add manual prefetch:

```zig
// In findAll() hot loop
if (comptime builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64) {
    @prefetch(&states[next_state], .{ .rw = .read, .locality = 3 });
}
```

**Estimated improvement**: +5-10% (if memory-bound)

---

## Success Criteria

| Metric | Baseline | Phase 2 Target | Phase 3 Stretch |
|--------|----------|----------------|-----------------|
| Throughput | 125.0 MB/sec | ≥160 MB/sec | ≥200 MB/sec |
| Memory | 66 KB (1000 patterns) | ≤70 KB | ≤80 KB |
| Tests passing | 722/722 | 722/722 | 722/722 |
| Code complexity | Low | Low | Medium |

**Decision criteria**:
- If Phase 2 achieves ≥180 MB/sec → **SHIP** (90% of 200 MB/sec target)
- If Phase 2 achieves 160-179 MB/sec → Proceed to Phase 3
- If Phase 3 achieves ≥200 MB/sec → **SHIP** (100% of target)
- If Phase 3 achieves 180-199 MB/sec → **SHIP** (90%+ acceptable, document trade-offs)

---

## Appendix: Cache Line Arithmetic

**Assumptions** (Apple M2 / AMD Ryzen 7):
- L1 cache: 64 KB, 64-byte lines, 4-way associative
- L1 latency: 4-5 cycles (~1.5 ns @ 3 GHz)
- L2 latency: 12-15 cycles (~5 ns)
- L3 latency: 40-60 cycles (~15-20 ns)
- DRAM latency: 200-300 cycles (~70-100 ns)
- Cache miss to DRAM: ~200 ns (measured in v1.6.0 RBTree analysis)

**Current layout** (separate arrays):
- State 0 data: BASE[0] in line A, CHECK[0] in line B, FAIL[0] in line C, OUTPUT[0] in line D
- **4 cache lines** to access one state's data
- If state access pattern is random → all 4 likely in different cache lines → 4 DRAM accesses

**Linearized layout** (SoA struct):
- State 0 data: All fields in 20 bytes → fits in **1 cache line**
- Random access → 1 DRAM access per state (vs 4 current)
- **4× reduction in cache misses**

**Expected speedup**: 4× reduction in misses ≈ **60-80% throughput improvement**
(Not linear due to CPU pipelining, instruction overhead, but significant)

---

## References

- v1.6.0 RedBlackTree Performance Analysis: Cache miss penalty ~200 ns
- v1.8.0 DoubleArrayTrie Implementation: Current sparse allocation strategy
- Aoe 1989: Original double-array trie paper (construction algorithm, not cache optimization)
- Intel Optimization Manual: Cache line size (64 bytes), prefetching strategies
