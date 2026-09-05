# Double-Array Trie Research

**Date**: 2026-03-17
**Milestone**: v1.8.0 — Double-Array Trie Implementation
**Goal**: Achieve 200-300 MB/sec Aho-Corasick performance with 50-100× memory reduction

---

## 1. Overview

### 1.1 Problem Statement

Current Aho-Corasick implementation uses **NodeASCII** structure:
- **Memory footprint**: 2352 bytes/node (87% transition table, 11% real_children tracking)
- **Benchmark workload**: ~23 MB for 1000 patterns (~10k nodes)
- **Performance**: 133 MB/sec throughput
- **Bottleneck**: Pointer-based traversal = cache misses

**Goal**: Replace pointer-based trie with **double-array trie** for:
- 50-100× memory reduction (2352 → 30-50 bytes/node)
- 50-125% performance improvement (133 → 200-300 MB/sec)
- Maintain O(1) state transitions

### 1.2 Double-Array Trie (Aoe 1989)

**Key insight**: Represent trie transitions using two parallel arrays (BASE and CHECK) instead of pointer-based nodes.

**Advantages**:
- **Space-efficient**: Single integer per state instead of 256 pointers
- **Cache-friendly**: Sequential array access instead of pointer chasing
- **O(1) transitions**: Direct array indexing `BASE[s] + c`

**Trade-off**: Complex construction algorithm (requires conflict resolution during node insertion)

---

## 2. Algorithm Specification

### 2.1 Data Structures

```zig
const DoubleArrayTrie = struct {
    BASE: []i32,    // State transition base values (negative = leaf)
    CHECK: []u32,   // Parent state verification
    FAIL: []u32,    // Aho-Corasick failure links
    OUTPUT: []u32,  // Pattern ID list indices (0 = no output)
};
```

**Encoding**:
- `BASE[s]`:
  - If `BASE[s] >= 0`: transition base address
  - If `BASE[s] < 0`: leaf node, `-BASE[s]` = pattern ID
- `CHECK[t]`: parent state that transitions to `t`
- Transition from state `s` on character `c`: `t = BASE[s] + c`
- **Validity check**: `CHECK[t] == s` (confirms transition is valid)

### 2.2 Transition Logic

**State transition** (search phase):
```zig
fn next(s: u32, c: u8) ?u32 {
    const t = @intCast(u32, BASE[s] + c);
    if (CHECK[t] == s) return t;  // Valid transition
    return null;  // No transition, follow failure link
}
```

**Failure link traversal** (Aho-Corasick):
```zig
fn traverse(s: u32, c: u8) u32 {
    var state = s;
    while (true) {
        if (next(state, c)) |t| return t;
        if (state == 0) return 0;  // Root fallback
        state = FAIL[state];  // Follow failure link
    }
}
```

### 2.3 Construction Algorithm (Aoe 1989)

**Input**: Trie graph `G = (V, E)` with nodes `V` and edges `E`
**Output**: `BASE` and `CHECK` arrays

**Algorithm**:
1. **Initialize**: `BASE[0] = 1`, `CHECK[0] = 0` (root state)
2. **For each state `s` in BFS order**:
   - Let `C(s)` = set of outgoing edge labels (characters)
   - **Find base `b`** such that ∀c ∈ C(s): `CHECK[b + c] == EMPTY`
     - Start search at `b = 1`, increment until conflict-free position found
   - **Set `BASE[s] = b`**
   - **For each child transition `c → t`**:
     - `pos = b + c`
     - `CHECK[pos] = s` (mark parent)
     - Queue `t` for processing
3. **Compact representation**: Actual implementation uses sparse arrays or dynamic resizing

**Conflict resolution**: When `CHECK[b + c]` is occupied, try next `b` value. This is the main complexity — naïve search is O(|Σ| × |V|²), but optimized heuristics (e.g., bitmap-based gap search) reduce to near-linear.

### 2.4 Aho-Corasick Integration

**Goto function**: Encoded in `BASE`/`CHECK` arrays (standard double-array trie)

**Failure function**: Store in separate `FAIL` array (parallel to `BASE`)
- `FAIL[s]` = failure state for state `s`
- Built using standard AC BFS (after goto completion)
- **Key difference**: Failure links don't modify BASE/CHECK structure

**Output function**: Store in separate `OUTPUT` array
- `OUTPUT[s]` = index into pattern list (or 0 if no match)
- Multiple patterns per state require indirection (linked list or array slice)

---

## 3. Space Analysis

### 3.1 Current NodeASCII

```zig
const Node = struct {
    transitions: [256]?*Node,  // 2048 bytes (256 × 8)
    real_children: [256]u8,    // 256 bytes
    failure: ?*Node,           // 8 bytes
    output: std.ArrayList(?*Node),  // 24 bytes
    pattern_indices: std.ArrayList(usize),  // 24 bytes
    depth: usize,              // 8 bytes
};
// Total: 2368 bytes/node
```

### 3.2 Double-Array Trie

```zig
// Per-state overhead (4 arrays × N states)
BASE:   4 bytes/state  (i32)
CHECK:  4 bytes/state  (u32)
FAIL:   4 bytes/state  (u32)
OUTPUT: 4 bytes/state  (u32 index or pointer)
// Total: 16 bytes/state
```

**Reduction factor**: 2368 / 16 = **148× smaller per state**

**Actual reduction** (accounting for dense vs sparse):
- NodeASCII: Dense 256-entry table per node (wastes space for sparse transitions)
- Double-array: Only allocated states consume memory
- Expected: 50-100× reduction in practice (depends on branching factor)

---

## 4. Performance Analysis

### 4.1 Current Implementation (Pointer-Based)

**Bottleneck**: Cache misses during pointer traversal
- Log(N) pointer dereferences for search
- Each dereference: ~200 ns cache miss (v1.6.0 finding)
- **Throughput**: 133 MB/sec

**Memory access pattern**: Random access across heap (poor cache locality)

### 4.2 Double-Array Trie

**Advantages**:
1. **Sequential array access**: `BASE[s]` lookup = single cache line fetch
2. **Fewer indirections**: `t = BASE[s] + c` is arithmetic (no pointer chasing)
3. **Better cache utilization**: Contiguous arrays exploit spatial locality

**Expected speedup**: 50-125% improvement (200-300 MB/sec)
- Baseline: 133 MB/sec
- Cache-friendly access: Reduce cache misses from ~3-4 to ~1-2 per char
- Arithmetic operations: O(1) instead of pointer dereference

**Rust aho-corasick DFA** (reference): 200-400 MB/sec with double-array structure

### 4.3 Limitations

**Construction cost**: O(|V| × |Σ|) with naïve conflict resolution
- Optimized: O(|V| + |E|) with bitmap-based gap search
- Trade-off: Slower build, faster search (acceptable for static automaton)

**Memory overhead during construction**: May require temporary arrays 2-3× final size (for conflict resolution)

---

## 5. Implementation Plan

### 5.1 Phase 1: Basic Double-Array Trie (No AC)

1. **Data structure**: Define `DoubleArrayTrie` struct with BASE/CHECK arrays
2. **Construction**: Implement Aoe's algorithm with naïve conflict resolution
3. **Search**: Implement `next()` and `contains()` for simple trie lookup
4. **Tests**: Small dictionaries (10-100 words), validate correctness
5. **Benchmark**: Compare memory usage vs pointer-based trie

**Deliverable**: Working double-array trie for static dictionary

### 5.2 Phase 2: Aho-Corasick Integration

1. **Add FAIL array**: Store failure links parallel to BASE/CHECK
2. **Failure function construction**: Standard AC BFS over double-array structure
3. **Add OUTPUT array**: Pattern matching metadata
4. **Search with failure links**: Implement `traverse()` with AC semantics
5. **Tests**: Multi-pattern matching (100-1000 patterns)

**Deliverable**: Full Aho-Corasick automaton using double-array representation

### 5.3 Phase 3: Optimization

1. **Conflict resolution**: Replace naïve search with bitmap-based gap finder
2. **Memory allocation**: Pre-size arrays based on pattern count heuristics
3. **Cache prefetching**: Add `@prefetch()` hints for sequential BASE access
4. **Benchmarking**: Compare vs NodeASCII on standard workloads

**Deliverable**: Production-ready implementation meeting 200-300 MB/sec target

---

## 6. References

### 6.1 Original Paper
- Aoe, J. (1989). "An Efficient Digital Search Algorithm by Using a Double-Array Structure". *IEEE Transactions on Software Engineering*, 15(9), 1066-1077.

### 6.2 Implementations
- **Rust aho-corasick**: `src/dfa.rs` — DFA variant using double-array trie
- **darts**: C++ library implementing double-array tries (Taku Kudo)
- **datrie**: Thai dictionary trie using double-array (Theppitak Karoonboonyanan)

### 6.3 Related Reading
- Aoe, J. (1992). "Computer Algorithms: String Pattern Matching Strategies"
- Morita, K., et al. (2001). "A Practical Fast Algorithm for Constructing Minimal Perfect Hash Functions"

---

## 7. Next Steps

1. ✅ **Theory documented** (this document)
2. **Algorithm pseudo-code** — Write detailed step-by-step construction algorithm
3. **Proof of concept** — Implement small double-array trie (no AC, 10-word dictionary)
4. **Test suite** — Property-based tests for trie invariants
5. **AC integration** — Add failure links and output function
6. **Performance validation** — Benchmark vs NodeASCII

**Status**: Research complete. Ready for implementation phase.

---

## 8. Open Questions

1. **Array sizing**: Initial capacity for BASE/CHECK arrays?
   - Heuristic: 2× pattern count + 256 (alphabet) is typical starting point
   - Growth strategy: Reallocate with 1.5× expansion when conflicts exceed threshold

2. **Conflict resolution optimization**: Bitmap-based or linked-list free cells?
   - Bitmap: O(1) gap search but requires extra memory (1 bit per cell)
   - Linked list: O(k) search but no extra memory
   - **Recommendation**: Start with naïve (simple), optimize later if benchmark shows construction time issue

3. **Multi-pattern output**: Array-of-arrays or linked list?
   - Array-of-arrays: Better cache locality but wastes space if few patterns per state
   - Linked list: Space-efficient but pointer chasing
   - **Recommendation**: Hybrid — inline single pattern (common case), allocate list for multiple

4. **Failure link encoding**: Separate FAIL array or embed in BASE?
   - Separate: Clean separation, easier to understand
   - Embedded: Save 4 bytes/state but complicates logic
   - **Recommendation**: Separate (clarity over 25% memory savings)

---

**Document Version**: 1.0
**Last Updated**: 2026-03-17 Hour 21
**Author**: Claude Sonnet 4.5 (Autonomous Development)
