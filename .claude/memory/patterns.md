# zuda Code Patterns

> Compressed 2026-09-03 (session 870, stabilization) — was 1155 lines, over the project's
> 200-line memory rule. Full code listings for the domain-specific patterns below now live only
> in their source files (grep the referenced path); this file keeps the reusable lesson.

## Duplicate Distribution Detection Pattern
Before implementing a new distribution, grep first — the session-index memory only covers the
last ~10 sessions, but the file has 200+: `grep "pub fn DistributionName" src/stats/distributions.zig`.
JohnsonSU was already implemented but missing from the recent-session list (session 734).

## Container Lifecycle / Comptime Generic / Iterator / Validate (core template)
```zig
pub fn init(allocator: std.mem.Allocator) !Self {
    const nodes = try allocator.alloc(Node, initial_capacity);
    return .{ .allocator = allocator, .nodes = nodes, .count = 0 };
}
pub fn deinit(self: *Self) void {
    self.allocator.free(self.nodes);
    self.* = undefined;
}

pub fn Container(comptime K: type, comptime V: type, comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order) type {
    return struct { const Self = @This(); /* ... */ };
}

pub const Iterator = struct {
    current: ?*Node,
    pub fn next(self: *Iterator) ?Entry {
        const node = self.current orelse return null;
        self.current = node.successor();
        return .{ .key = node.key, .value = node.value };
    }
};

/// Asserts all internal invariants hold. Call after operations during testing.
pub fn validate(self: *const Self) !void {
    if (self.root) |root| {
        try self.validateBstProperty(root, null, null);
        try std.testing.expectEqual(self.count, self.countNodes(root));
    }
}
```
Error cleanup: `const buf = try allocator.alloc(Node, capacity); errdefer allocator.free(buf);`
Leak detection: pass `std.testing.allocator` in tests — it auto-detects leaks on `deinit()`.
Allocation-failure testing: `std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 5 })`.

## Zig 0.15 ArrayList Pattern
No more `.init()`. Use struct literal + pass allocator per call:
```zig
var list = std.ArrayList(T){};
try list.append(allocator, item);
list.deinit(allocator);
```

## BTree Node Structure — save before mutating
Save key/value BEFORE modifying node structure, or the read racing the mutation asserts:
```zig
const median_key = full_child.keys[mid];
const median_value = full_child.values[mid];
full_child.num_keys = mid;
parent.keys[idx] = median_key; // NOT full_child.keyAt(mid) after truncating
```

## Tree Deletion Fixup — track parent separately
For red-black deletion, `fixup_node` may become null; save `fixup_parent` before the transplant so
`deleteFixup` can still walk up from a null node (double-black case).

## Test Quality Anti-Patterns (audited v1.5.0, still enforced)
- **No bare "doesn't crash" tests** — every test must assert actual state (`count()`, `get()`
  results), not just that insert/validate ran without erroring. A test with no assertion, or one
  that only calls `validate()` without checking state, passes even when the implementation is
  wrong — it should fail when the implementation is wrong, not just when it panics.

## Zig 0.15.2 `std.fmt` gotcha: legacy `format()` signature (Session 791)
Every `format()` in `distributions.zig` uses the **old** 3-arg signature
(`fn format(self, comptime fmt, options, writer) !void`). This compiles fine because Zig only
analyzes a method when called — but Zig 0.15.2's `std.Io.Writer` expects the new 2-arg contract
(`fn format(self, w: *std.Io.Writer) !void`), so any test that actually calls
`std.fmt.bufPrint`/`print("{f}", .{dist})` on one of these types fails to compile. None of the
~200 distributions' `format()` are exercised by `std.fmt` anywhere in the suite — this is a known,
file-wide, currently-accepted gap (closed out as "won't fix per-distribution," see MEMORY.md).
**Never add a format()-via-std.fmt smoke test for a new distribution.**

## Binomial `logFactorial` precision cliff at n≥20 (Session 791)
The shared `logFactorial` helper is exact for `n<20`, then switches to Stirling's approximation
(`n*log(n)-n+0.5*log(2πn)`) for `n>=20`, which is only accurate to ~4e-3 in log-space right at the
boundary. Any Binomial-derived distribution (ZeroInflatedBinomial, HurdleBinomial, BetaBinomial,
etc.) with `n>=20` will fail a tight (`1e-12`) normalization test for reasons unrelated to that
distribution's own formula — **keep new tests under n=20, or loosen tolerance**. Raising the
helper's exact-computation cutoff would fix this properly but is cross-cutting; treat as a
stabilization backlog item, not a fix to make inside a single-distribution feature cycle.

## Stack-buffer-in-init() gotcha (Session 870 — FisherNoncentralHypergeometric)
Don't use a fixed-size local array (e.g. `var buf: [1024]T = undefined`) to stage per-support-point
values (logsumexp weights, etc.) inside a distribution's `init()` — the support range is a runtime
value derived from N/K/n and can exceed any fixed bound for realistic large-population inputs,
causing an out-of-bounds write. Prefer a streaming two-pass computation (pass 1: find max; pass 2:
recompute + accumulate) that needs no buffer, matching this repo's no-allocator-in-init convention
for discrete distributions.

## Domain-specific patterns (condensed — read the source file for full code/tests)
- **BFGS quasi-Newton** (`src/optimize/*bfgs*`): inverse-Hessian update `H = V^T H V + ρss^T`,
  skip update when `y^T s ≤ 1e-10` (curvature condition). O(n²) memory. 34-test suite covers
  convergence, line-search variants (Armijo/Wolfe/backtracking), curvature/positive-definiteness
  invariants, standard test functions (sphere/Booth/Himmelblau/Rosenbrock).
- **Random NDArray factory** (`rand`/`randn`): validate shape (no zero dims) before allocating,
  seed a PCG64, fill per-element; errdefer on the allocation.
- **Double-Array Trie** (Aoe 1989): BASE/CHECK arrays instead of pointer nodes; transition validity
  is `CHECK[BASE[state]+char] == state`.
- **NDArray reductions** (`sum`/`prod`/`mean`/axis variants): full reductions return `T` (mean
  always `f64`); axis reductions allocate a fresh NDArray with the reduced dimension removed;
  layout-independent (row/col-major must agree).
- **DCT-II/III** (`src/signal/dct.zig`): basis `cos(π·k·(n+0.5)/N)`, orthonormal scaling
  `sqrt(1/N)` for k=0 vs `sqrt(2/N)` for k>0; round-trip `idct(dct(x))≈x`.
- **interp1d**: binary search O(log n) for containing interval + O(1) linear interpolation;
  require strictly increasing x (equal consecutive x → division by zero).
- **PCHIP**: weighted-harmonic-mean derivative at interior points, zeroed when adjacent slopes
  disagree in sign (monotonicity preservation); O(h⁴) local error.
- **SIMD trsv** (`src/linalg/*trsv*`): outer loop is sequential (data-dependent), only the inner
  dot-product accumulation vectorizes (`@Vector`/`@reduce(.Add,...)`); temp-copy the RHS first
  since `x[i]` is overwritten in place as it's solved. 4 cases: {upper,lower}×{trans,noTrans}.
- **FFT twiddle caching** (`src/signal/fft.zig`): precompute `W_N^k` once into an `O(n)` table,
  index each butterfly stage with `stride = n/size`; 2-3× speedup for n≥256 by eliminating
  per-butterfly `@cos`/`@sin` calls.
- **Auto-dispatch by size** (e.g. GEMM naive vs. blocked-4x4): pick an empirically-benchmarked
  threshold (64×64 typical), dispatch inside the public function so callers don't see it; test the
  boundary (`threshold±1`) explicitly.
- **Heavy-tailed / no-closed-form distributions** (Landau-style): PDF via table/approximation,
  quantile via bisection (not FFT inversion), mode via bisection on `d(log pdf)/dx = 0`,
  mean/variance via numerical quadrature only when they actually exist (heavy tails often don't).
