# zuda Code Patterns

## Duplicate Distribution Detection Pattern (Session 734)
Before implementing a new distribution, always grep for an existing implementation:
```bash
grep "pub fn DistributionName" src/stats/distributions.zig
```
JohnsonSU was already at line 38522 but not in the MEMORY.md "latest session" list — the list only goes back ~10 sessions. Always grep the file, not just memory.

## Borel Distribution Sampling Pattern (Session 734)
For discrete distributions with power-law-like tails (Borel, Zeta, Zipf), use linear scan inverse CDF
rather than binary search — it avoids precomputing the CDF table and is simpler for unbounded support:
```zig
pub fn quantile(self: Self, p: T) DistributionError!u64 {
    if (!(p >= 0.0 and p <= 1.0)) return error.InvalidProbability;
    if (p == 0.0) return 1;
    var cumsum: T = 0.0;
    var k: u64 = 1;
    while (k <= MAX_K) : (k += 1) {
        cumsum += self.pmf(k);
        if (cumsum >= p) return k;
    }
    return MAX_K;
}
```
Use MAX_K = 50000 as a safety cap. Break early when p is small (most samples hit k=1 quickly for μ≪1).

## Random NDArray Factory Pattern (Session 528)
Create random arrays with seeded PRNGs for reproducibility:

```zig
pub fn rand(allocator: Allocator, shape: []const usize, seed: u64, layout: Layout) !Self {
    // 1. Validate shape (no zero dimensions)
    for (shape) |dim| {
        if (dim == 0) return error.ZeroDimension;
    }

    // 2. Create array structure
    var arr = try Self.init(allocator, shape, layout);
    errdefer arr.deinit();

    // 3. Initialize PRNG with seed
    const random_module = @import("../stats/random.zig");
    var rng = random_module.Pcg64.init(seed);

    // 4. Fill with random values
    for (arr.data) |*val| {
        val.* = rng.random().float(T);  // Uniform [0, 1) for rand()
        // OR
        val.* = random_module.normal(T, rng.random());  // N(0,1) for randn()
    }

    return arr;
}
```

**Key Points**:
- Validate before allocation (fail fast)
- Use errdefer for cleanup on error paths
- Seed-based reproducibility (same seed → same array)
- Generic over float types (f32, f64)
- Respects layout (row-major/column-major)

## Quasi-Newton Optimization Pattern (BFGS)
Implement approximate Hessian (inverse) iteratively to accelerate convergence:

```zig
pub fn bfgs(comptime T: type, f, grad_f, x0, options, allocator) !OptimizationResult(T) {
    // 1. Initialize H = I (identity matrix)
    const H = try allocator.alloc(T, n * n);
    for (0..n) |i| {
        for (0..n) |j| {
            H[i * n + j] = if (i == j) @as(T, 1) else @as(T, 0);
        }
    }

    // 2. Each iteration:
    //    a. p = -H * grad  (search direction)
    //    b. Line search for α
    //    c. x_new = x + α * p
    //    d. s = x_new - x,  y = grad_new - grad_old
    //    e. Update: H = V^T * H * V + ρ*s*s^T  where V = I - ρ*s*y^T
    //       (only if y^T*s > epsilon for curvature)

    // Matrix-vector multiply: p = -H * grad
    for (0..n) |i| {
        var sum: T = 0;
        for (0..n) |j| sum += H[i * n + j] * grad[j];
        p[i] = -sum;
    }

    // BFGS Hessian update formula
    var y_dot_s: T = 0;
    for (0..n) |i| y_dot_s += y[i] * s[i];

    if (y_dot_s > 1e-10) {  // Curvature condition check
        const rho = 1.0 / y_dot_s;

        // V = I - ρ*s*y^T
        var V = try allocator.alloc(T, n * n);
        defer allocator.free(V);
        for (0..n) |i| {
            for (0..n) |j| {
                const delta = if (i == j) @as(T, 1) else @as(T, 0);
                V[i * n + j] = delta - rho * s[i] * y[j];
            }
        }

        // Temp = H * V
        var Temp = try allocator.alloc(T, n * n);
        defer allocator.free(Temp);
        for (0..n) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                for (0..n) |k| sum += H[i * n + k] * V[k * n + j];
                Temp[i * n + j] = sum;
            }
        }

        // H = V^T * Temp + ρ*s*s^T
        for (0..n) |i| {
            for (0..n) |j| {
                var sum: T = 0;
                for (0..n) |k| sum += V[k * n + i] * Temp[k * n + j];
                H[i * n + j] = sum + rho * s[i] * s[j];
            }
        }
    }
}
```

**Key insights**:
- Hessian stays symmetric positive definite if Wolfe line search is used
- Skip update if y_k^T * s_k is too small (numerical stability)
- O(n²) memory required for the inverse Hessian approximation
- Superlinear convergence on strongly convex functions

## Container Lifecycle Pattern
```zig
pub fn init(allocator: std.mem.Allocator) !Self {
    const nodes = try allocator.alloc(Node, initial_capacity);
    return .{ .allocator = allocator, .nodes = nodes, .count = 0 };
}

pub fn deinit(self: *Self) void {
    self.allocator.free(self.nodes);
    self.* = undefined;
}

// Caller:
var tree = try RedBlackTree(i64, void).init(allocator);
defer tree.deinit();
```

## Comptime Generic Pattern
```zig
pub fn RedBlackTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type {
    return struct {
        const Self = @This();
        // ...
    };
}
```

## Iterator Pattern
```zig
pub const Iterator = struct {
    current: ?*Node,

    pub fn next(self: *Iterator) ?Entry {
        const node = self.current orelse return null;
        self.current = node.successor();
        return .{ .key = node.key, .value = node.value };
    }
};

pub fn iterator(self: *const Self) Iterator {
    return .{ .current = self.minimum() };
}
```

## Invariant Validation Pattern
```zig
/// Asserts all internal invariants hold. Call after operations during testing.
pub fn validate(self: *const Self) !void {
    if (self.root) |root| {
        // BST property
        try self.validateBstProperty(root, null, null);
        // Color invariant (for RB trees)
        try self.validateColorInvariant(root);
        // Count matches
        try std.testing.expectEqual(self.count, self.countNodes(root));
    }
}
```

## Test with Leak Detection
```zig
test "no memory leaks" {
    const allocator = std.testing.allocator; // auto-detects leaks
    var tree = try RedBlackTree(i64, void).init(allocator);
    defer tree.deinit();
    // operations...
}
```

## Error Cleanup Pattern
```zig
const buf = try allocator.alloc(Node, capacity);
errdefer allocator.free(buf);
// If subsequent operations fail, buf is freed
```

## Allocation Failure Testing
```zig
test "handles allocation failure" {
    var failing = std.testing.FailingAllocator.init(
        std.testing.allocator,
        .{ .fail_index = 5 },
    );
    // Operations that allocate will fail after 5th allocation
}
```

## Double-Array Trie Pattern (Aoe 1989)
Space-efficient trie using BASE/CHECK arrays instead of pointer nodes:

```zig
pub fn DoubleArrayTrie(comptime T: type) type {
    return struct {
        base: []i32,        // State transition base (or next unallocated ID)
        check: []u32,       // Parent state verification (0xFFFFFFFF = empty)
        is_leaf: []bool,    // Pattern endings (separate array)
        state_count: u32,
        allocator: Allocator,
        patterns: []const []const T,

        // Transition: pos = BASE[state] + char; valid if CHECK[pos] == state
        pub fn contains(self: *const Self, key: []const T) bool {
            var state: u32 = 0;
            for (key) |char| {
                const pos = @as(u32, @intCast(self.base[state] + @as(i32, @intCast(@as(u8, @intCast(char))))));
                if (pos >= self.check.len or self.check[pos] != state) return false;
                state = pos;
            }
            return state < self.is_leaf.len and self.is_leaf[state];
        }
    };
}
```

Construction: Reserve 256 positions per state for simplicity. More complex implementations use conflict resolution to minimize space.

## Zig 0.15 ArrayList Pattern
In Zig 0.15, ArrayList no longer has `.init()` method. Use struct literal instead:

```zig
// OLD (0.13): var list = std.ArrayList(T).init(allocator);
// NEW (0.15):
var list = std.ArrayList(T){};

// Methods now take allocator as first parameter:
try list.append(allocator, item);
list.deinit(allocator);
```

## BTree Node Structure
For B-trees, save key/value BEFORE modifying node structure:

```zig
// WRONG:
full_child.num_keys = mid;
parent.keys[idx] = full_child.keyAt(mid); // assertion fails!

// RIGHT:
const median_key = full_child.keys[mid];
const median_value = full_child.values[mid];
full_child.num_keys = mid;
parent.keys[idx] = median_key;
```

## In-Order Iterator for Multi-Child Trees
For B-trees with multiple keys per node, interleave children and keys:

```zig
// For node with keys [k0, k1, k2] and children [c0, c1, c2, c3]:
// Sequence: c0, k0, c1, k1, c2, k2, c3
if (!node.is_leaf) {
    if (idx % 2 == 0) {
        // Even: descend to child[idx/2]
        descend(child[idx/2]);
    } else {
        // Odd: return key[idx/2]
        return key[idx/2];
    }
}
```

## Tree Deletion with Fixup Parent Tracking
For red-black tree deletion, track parent separately when fixup node may be null:

```zig
fn deleteNode(self: *Self, node: *Node) void {
    var original_color = node.color;
    var fixup_node: ?*Node = null;
    var fixup_parent: ?*Node = null;  // Track parent separately!

    if (node.left == null) {
        fixup_node = node.right;
        fixup_parent = node.parent;  // Save before transplant
        self.transplant(node, node.right);
    }
    // ... rest of deletion ...

    if (original_color == .black) {
        // Pass both node AND parent to fixup
        self.deleteFixup(fixup_node, fixup_parent);
    }
}

fn deleteFixup(self: *Self, node: ?*Node, parent: ?*Node) void {
    // Can now access parent even when node is null
    const p = parent orelse return;
    // ...
}
```

This pattern is essential when fixup may need to walk up the tree starting from a null node (double-black situation).

## Test Quality Anti-Patterns

**AVOID: Tests with no assertions** (discovered 2026-03-16 during v1.5.0 quality audit)

```zig
// BAD: Just calls methods, doesn't verify behavior
test "memory leak test" {
    var map = try CuckooHashMap(u32, u32).init(allocator, .{});
    defer map.deinit();

    for (0..100) |i| {
        _ = try map.insert(i, i);
    }
    // No assertions! Only implicit leak check from allocator
}

// GOOD: Explicitly verify expected state
test "memory leak test" {
    var map = try CuckooHashMap(u32, u32).init(allocator, .{});
    defer map.deinit();

    for (0..100) |i| {
        _ = try map.insert(i, i);
    }
    try testing.expectEqual(@as(usize, 100), map.count());

    // Verify keys are accessible
    for (0..100) |i| {
        try testing.expectEqual(@as(?u32, i), map.get(i));
    }
}
```

**AVOID: Validate-only tests without state verification**

```zig
// BAD: Just checks validate() doesn't error
test "validate invariants" {
    var map = try HashMap(u32, u32).init(allocator, .{});
    defer map.deinit();

    try map.validate();
    _ = try map.insert(1, 100);
    try map.validate();  // What should the state be?
}

// GOOD: Assert expected state at each step
test "validate invariants" {
    var map = try HashMap(u32, u32).init(allocator, .{});
    defer map.deinit();

    try map.validate();
    try testing.expectEqual(@as(usize, 0), map.count());

    _ = try map.insert(1, 100);
    try map.validate();
    try testing.expectEqual(@as(usize, 1), map.count());
    try testing.expectEqual(@as(?u32, 100), map.get(1));
}
```

**Key principle**: Every test must verify actual behavior, not just "doesn't crash". Tests should fail when the implementation is wrong, not just when it panics.

## NDArray Reduction Operation Testing Pattern (2026-03-21)

For multi-dimensional reductions (sum, prod, mean, min, max), test structure:

```zig
// Pattern for full reductions (returns scalar T)
test "reduction: sum() full 1D array i32" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set data
    for (0..5) |i| {
        arr.data[i] = @intCast(i + 1);  // [1, 2, 3, 4, 5]
    }

    // Test assertion
    const result = try arr.sum();
    try testing.expectEqual(@as(i32, 15), result);  // 1+2+3+4+5
}

// Pattern for axis reductions (returns NDArray with reduced shape)
test "reduction: sum() axis 0 on 2D array [3,4]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set data: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Test axis reduction
    const result = try arr.sumAxis(allocator, 0);  // [3,4] -> [4]
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    // Result: [1+5+9=15, 2+6+10=18, 3+7+11=21, 4+8+12=24]
    try testing.expectEqual(@as(i32, 15), result.data[0]);
    try testing.expectEqual(@as(i32, 18), result.data[1]);
}

// Key testing principles for reductions:
// 1. Full reductions: verify scalar output equals mathematical result
// 2. Axis reductions: verify shape is correct (remove reduced dimension)
// 3. Use sequential fill (0..n) for predictable hand-computed sums
// 4. For floats, use testing.expectApproxEqAbs() with epsilon 1e-10
// 5. Test all axes on multi-dimensional arrays (axis 0, 1, 2 for 3D, etc.)
// 6. Edge cases: negative values, zeros, single element, layout variation
// 7. Type conversions: mean() always returns f64 even for i32 arrays
// 8. Error paths: axis out of bounds, empty arrays (if disallowed)
```

**Design decisions**:
- Full reductions: sum/prod/min/max return `T`, mean always returns `f64`
- Axis reductions: Return new NDArray with removed dimension (allocate fresh)
- Layout independence: Results same regardless of row/column-major
- Type safety: Integer operations stay in type T, mean promotes to f64
- Error handling: IndexOutOfBounds for invalid axis, (TBD) ZeroDimension or similar for empty

## Beta Distribution Test Pattern (2026-03-22)

Reference implementation: `src/stats/distributions/beta.zig` (skeleton with 53 tests)

**Test Structure** (adapted from Gamma/Normal pattern):
1. **init tests (6)**: Valid params, error cases (alpha/beta ≤ 0)
2. **pdf tests (11)**: Outside support → 0, boundary behavior (x=0, x=1), mode verification, symmetry property, normalization
3. **cdf tests (10)**: F(0)=0, F(1)=1, monotonicity, special cases (Uniform), bounds [0,1]
4. **quantile tests (10)**: Q(0)=0, Q(1)=1, inverse property |cdf(quantile(p))-p|<ε, monotonicity, error handling
5. **logpdf tests (5)**: log(pdf) consistency, -∞ outside support, numerical stability
6. **sample tests (10)**: Range [0,1], mean/variance statistical convergence (10k samples, 3-5% tolerance), edge cases
7. **integration tests (5)**: PDF normalization via trapezoid rule, CDF-quantile inverse, ensemble statistics

**Key Test Patterns**:
- Use `expectApproxEqAbs` for equality checks (pdf/cdf values, statistical means)
- Use `expectApproxEqRel` for relative error checks (variance with smaller absolute values)
- Statistical tests: 10,000 samples with 3-5% tolerance for mean, 10% for variance
- Boundary checks: test at 0.0, 1.0, and outside [0,1]
- Special case validation: Beta(1,1)=Uniform, symmetry Beta(α,β)↔Beta(β,α)

**RNG Pattern**:
```zig
var prng = std.Random.DefaultPrng.init(seed);
const rng = prng.random();
const sample = dist.sample(rng);  // NOT &rng.random()
```

**Failed Test Insights** (to guide implementation):
- Quantile requires accurate CDF inversion (Newton-Raphson sensitive to initial guess and convergence)
- Incomplete beta function needs higher precision than naive series (use continued fractions or more terms)
- Beta function log: logB(α,β) = logΓ(α) + logΓ(β) - logΓ(α+β), watch for overflow/underflow
- Edge case Beta(1,1) must be exactly Uniform: pdf(x)=1.0, cdf(x)=x, quantile(p)=p

## Signal Processing Transform Pattern (2026-03-24)

DCT Type II/III implementation in `src/signal/dct.zig`:

```zig
pub fn dct(comptime T: type, signal: []const T, allocator: Allocator) Allocator.Error![]T {
    // 1. Allocate output and handle errors
    const n = signal.len;
    const coeffs = try allocator.alloc(T, n);
    errdefer allocator.free(coeffs);

    if (n == 0) return coeffs;

    // 2. Pre-compute constants (n_f, π) for type T
    const n_f = @as(T, @floatFromInt(n));
    const pi = std.math.pi;

    // 3. Double loop: O(N²) naive computation
    for (0..n) |k| {
        var sum: T = 0.0;
        const k_f = @as(T, @floatFromInt(k));

        for (0..n) |n_idx| {
            const n_f_idx = @as(T, @floatFromInt(n_idx));
            // Core basis function: cos(π * k * (n + 0.5) / N)
            const angle = pi * k_f * (n_f_idx + 0.5) / n_f;
            sum += signal[n_idx] * @cos(angle);
        }

        // 4. Apply orthonormal scaling for energy conservation
        const scale = if (k == 0)
            @sqrt(1.0 / n_f)      // DC: sqrt(1/N)
        else
            @sqrt(2.0 / n_f);     // AC: sqrt(2/N)

        coeffs[k] = sum * scale;
    }

    return coeffs;
}
```

**Key patterns**:
- **Floating-point generics**: Use `comptime T: type` to support f32/f64. Cast integers to T for angle computation.
- **Pre-compute constants**: `n_f`, `π` once, not per iteration
- **Basis function**: `cos(π * k * (n + 0.5) / N)` is the DCT-II basis (note: (n+0.5), not n)
- **Orthonormal scaling**: Different for k=0 (DC) vs k>0 (AC) to ensure energy conservation
- **Inverse (Type III)**: Same pattern with coefficients input; same scaling ensures idct(dct(x)) ≈ x
- **Memory safety**: `errdefer allocator.free(coeffs)` to handle allocation failures
- **Empty signal**: Return empty array (valid edge case)

**Mathematical properties verified by tests**:
- Energy conservation: sum(dct(x)²) ≈ sum(x²) when orthonormal scaling applied
- DC component: coeffs[0] = sum(signal) * sqrt(1/N)
- Round-trip: idct(dct(x)) ≈ x within float precision (f64: 1e-9, f32: 1e-5)
- Linearity: dct(a*x + b*y) = a*dct(x) + b*dct(y)

**Test coverage strategy**:
- Basic: empty, single element, constant signal (energy at DC)
- Round-trip: various sizes including non-power-of-2
- Properties: energy conservation, DC value, orthogonality, linearity
- Edge cases: zero signal, negative values, large/small magnitudes, alternating (high-frequency)
- Types: both f32 and f64
- Memory: allocation/deallocation correctness
- Important: Remove duplicate defer statements (cause double-free in loops with reassignment)

## Numerical Interpolation Pattern (interp1d) — Session 18

**Algorithm**: 1D linear interpolation with constant extrapolation
- Input: x (sorted sample points), y (function values), x_new (query points)
- Output: interpolated y values at x_new
- Binary search to find containing interval: O(log n) per query
- Linear interpolation in interval: O(1)
- Total: O(m log n + m) where m = queries, n = samples

**Implementation structure**:
```zig
pub fn interp1d(comptime T: type, x: []const T, y: []const T, x_new: []const T, allocator: Allocator) ![]T {
    // Validate: x.len == y.len, x.len >= 2, x strictly increasing
    // Allocate result array
    // For each query point:
    //   - Binary search to find interval [x[j], x[j+1]] containing x_new[i]
    //   - If below min: result[i] = y[0]
    //   - If above max: result[i] = y[n-1]
    //   - Else: linear interpolation y0 + (y1-y0) * (x_new[i] - x0) / (x1 - x0)
    // Return allocated array
}
```

**Key validation**:
1. Dimension check: `x.len == y.len` → error.DimensionMismatch
2. Minimum length: `x.len >= 2` → error.InsufficientPoints
3. Monotonicity: `x[i] < x[i+1]` (strict inequality) → error.NonMonotonicX
   - Note: Equal consecutive values cause division by zero in interpolation

**Binary search pattern**:
```zig
var left: usize = 0;
var right: usize = x.len - 1;
while (left < right) {
    const mid = left + (right - left) / 2;
    if (x[mid] < xi) left = mid + 1;
    else right = mid;
}
// Result: x[left-1] < xi <= x[left], so interval is [left-1, left]
```

**Test strategy**:
1. **Exact cases**: Linear functions (interpolation is exact)
2. **Approximation**: Quadratic/exponential (verify approximation quality)
3. **Extrapolation**: Below/above domain (verify constant clamping)
4. **Edge cases**: Empty queries, single interval, constant function
5. **Non-uniform grids**: Irregular spacing tests
6. **Type support**: f32 (1e-5), f64 (1e-10) tolerances
7. **Large datasets**: 1000+ sample points, 500+ queries
8. **Error paths**: All 3 error types with dedicated tests
9. **Memory**: Caller ownership, allocation safety

**26 tests total**:
- 5 core functionality (exact, midpoint, linear, quadratic, 2-point)
- 3 extrapolation (below, above, mixed)
- 5 edge cases (empty, unsorted queries, constant, non-uniform, boundaries)
- 4 error handling (dimension, length, monotonicity, validation)
- 2 type support (f32, f64)
- 1 large dataset (1001 sample, 500 query)
- 1 memory (caller owns result)
- 4 numeric properties (exp, negative, small spacing, large spacing)

**Anti-patterns avoided**:
- Don't assume input is sorted (binary search finds correct interval)
- Don't re-allocate on each query (allocate once, fill in loop)
- Don't skip boundary validation (prevents crashes and incorrect results)
- Don't use simple loop search (O(m*n) vs O(m log n) — binary search mandatory)

**Numerical stability**:
- Small spacing (1e-6): test for underflow in (x_new[i] - x[j]) / dx
- Large spacing (1e6): test for overflow/precision loss
- Division dx = x[j+1] - x[j]: never zero due to strict monotonicity validation
- Linear combination: numerically stable (single multiplication/addition)


## PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) Pattern

**When to use**: Shape-preserving interpolation for smooth, monotonic data.

**Algorithm signature**:
```zig
pub fn pchip(comptime T: type, x: []const T, y: []const T, x_new: []const T, allocator: Allocator) ![]T
```

**Key implementation points**:
1. **Slope computation**: Compute differences δ[i] = (y[i+1] - y[i]) / h[i]
2. **Derivative computation**: Use weighted harmonic mean at interior points
   - w1 = 2*h[i] + h[i-1]
   - w2 = h[i] + 2*h[i-1]
   - d[i] = 2 / (w1/δ[i-1] + w2/δ[i])  [if slopes same sign, else 0]
3. **Monotonicity preservation**: Set d[i]=0 if δ[i-1] and δ[i] have opposite signs or either is zero
4. **Cubic Hermite interpolation**: Use basis functions h00, h10, h01, h11
5. **Constant extrapolation**: Return y[0] for x<x[0], y[n-1] for x>x[n-1]

**Accuracy expectations**:
- O(h⁴) local error between knots
- For smooth functions, needs ~5-10 points per domain unit for <1% relative error
- Exact for linear and constant functions
- Quadratic: ~3-5% error with 5 points over [0,1]

**Numerical stability**:
- Check denominator ≠ 0 in harmonic mean formula
- Check denominator != denominator (NaN detection) 
- Use careful handling of zero slopes
- Binary search for interval location O(log n)

**Testing considerations**:
- Verify monotonicity preservation on monotonic input
- Check C¹ continuity (smoothness at knots)
- Test exact values at sample points
- Test extrapolation behavior
- Be realistic about accuracy (3-5% error for quadratic with 5 points, not 1%)
- Ensure test grid actually contains sample points if testing "exact reproduction"

## Quasi-Newton (BFGS) Optimization Test Pattern — Session 41

**Algorithm structure**: BFGS maintains inverse Hessian approximation H that improves each iteration

```zig
pub fn bfgs(comptime T: type, f: ObjectiveFn(T), grad_f: GradientFn(T),
            x0: []const T, options: BfgsOptions(T),
            allocator: Allocator) !OptimizationResult(T)
```

**Options structure**:
```zig
pub fn BfgsOptions(comptime T: type) type {
    return struct {
        max_iter: usize = 1000,       // Iteration limit
        tol: T = 1e-6,                // Gradient norm tolerance
        line_search: LineSearchType = .wolfe,  // armijo|wolfe|backtracking
        ls_c1: T = 1e-4,              // Armijo constant
        ls_c2: T = 0.9,               // Curvature constant (Wolfe only)
        ls_max_iter: usize = 20,      // Line search iterations
    };
}
```

**Core iteration**:
1. Search direction: `p_k = -H_k * ∇f(x_k)` (H is inverse Hessian)
2. Line search: find α_k satisfying descent condition
3. Position update: `x_{k+1} = x_k + α_k * p_k`
4. Gradient update: compute ∇f(x_{k+1})
5. Compute: `s_k = α_k * p_k`, `y_k = ∇f(x_{k+1}) - ∇f(x_k)`
6. Verify: `y_k^T * s_k > 0` (curvature condition)
7. BFGS update:
   ```
   ρ_k = 1 / (y_k^T * s_k)
   H_{k+1} = (I - ρ_k * s_k * y_k^T) * H_k * (I - ρ_k * y_k * s_k^T) + ρ_k * s_k * s_k^T
   ```

**Test coverage** (34 tests across 8 categories):

1. **Basic Convergence (6 tests)**: Simple quadratic, 2D sphere, Rosenbrock, 5D, early termination, Beale
   - Each tests convergence to known minimum
   - Verify converged flag, final function value, gradient norm

2. **Line Search Variants (6 tests)**: Armijo, Wolfe, backtracking, comparisons, parameter effects, validation
   - Verify each line search method achieves descent
   - Compare convergence rates (Wolfe typically fastest)
   - Test parameter validation (c1 ∈ (0,1), c1 < c2 < 1)

3. **BFGS Properties (6 tests)**: First iteration behavior, Hessian improvement, superlinear convergence, curvature, positive definiteness, descent direction
   - First iteration should behave like gradient descent (H_0 = I)
   - Hessian approximation improves over iterations
   - Search direction p_k·∇f(x_k) < 0 maintained
   - y_k^T·s_k > 0 curvature condition validated
   - Superlinear convergence for strongly convex functions

4. **Convergence Properties (5 tests)**: Gradient/function monotonicity, convergence flag, max iterations, tolerance effects
   - Gradient norm ≤ previous iteration (monotonic)
   - Function value decreases (descent property)
   - Converged → ||∇f|| < tol
   - n_iter ≤ max_iter always
   - Tighter tolerance → more iterations

5. **Standard Test Functions (4 tests)**: Sphere, Booth, Himmelblau, minima verification
   - Sphere: convex, minimum at origin with f=0
   - Booth: minimum at (1,3) with f=0
   - Himmelblau: 4 minima, test starting near one
   - Verify final value within tolerance of known optimum

6. **Error Handling (3 tests)**: Empty x0, invalid parameters, negative tolerance
   - Empty x0 → error.InvalidArgument
   - Invalid line search (c1 ≥ c2, etc.) → error.InvalidArgument
   - tol ≤ 0 → error.InvalidArgument

7. **Type Support (2 tests)**: f32 (looser tolerance 1e-4), f64 (tight tolerance 1e-10)
   - Both types compile and converge
   - Type-appropriate tolerances

8. **Memory Safety (2 tests)**: No leaks (std.testing.allocator), independent calls
   - Multiple calls produce same result (no state sharing)
   - Allocator detects any leaks automatically

**Key testing principles**:
- Use meaningful test functions (sphere, Rosenbrock, Booth, Himmelblau, Beale)
- Test both success and error paths
- Verify invariants: descent direction, curvature condition, positive definiteness
- Compare convergence rates between line search methods
- Use realistic tolerances (1e-6 for f64, 1e-4 for f32)
- All tests should FAIL until implementation complete (TDD Red phase)
- Use placeholder `try testing.expect(false)` to ensure failure

**Anti-patterns to avoid**:
- Don't test Hessian approximation without H allocation in implementation
- Don't skip curvature condition validation (y^T·s ≤ 0 breaks algorithm)
- Don't assume faster convergence than gradient descent (BFGS is quasi-Newton, not Newton)
- Don't mix line search parameter validation with convergence tests
- Don't expect exact minima (numerical error acceptable)

## Performance Optimization Pattern: Auto-Dispatch to Specialized Implementations

When you have both generic and optimized implementations, use threshold-based auto-dispatch:

```zig
// Example: GEMM with naive vs blocked kernel
pub fn gemm(comptime T: type, alpha: T, A: NDArray(T, 2), B: NDArray(T, 2), beta: T, C: *NDArray(T, 2)) !void {
    // 1. Validate inputs (dimension checks, etc.)
    const m = A.shape[0];
    const n = B.shape[1];
    
    // 2. Choose implementation based on problem size
    const threshold: usize = 64;
    if (m >= threshold and n >= threshold) {
        // Large matrices → use cache-optimized blocked kernel
        return specialized_impl.gemm_blocked_4x4(T, alpha, A, B, beta, C);
    }
    
    // 3. Small matrices → use simple implementation (lower overhead)
    // ... naive triple-loop implementation ...
}
```

**When to use**:
- Multiple implementations with different performance characteristics
- Size-dependent performance tradeoffs (e.g., SIMD overhead vs speedup)
- One implementation is complex/optimized (blocking, SIMD), another is simple/low-overhead

**Guidelines**:
- Choose threshold empirically (benchmark crossover point)
- For BLAS: 64×64 is typical for blocking to outperform naive
- Preserve API compatibility — caller shouldn't know about dispatch
- Test boundary conditions (threshold-1, threshold, threshold+1)

**Testing Strategy** (TDD):
1. test-writer creates RED tests for dispatcher logic:
   - Boundary tests (size = threshold ± 1)
   - Correctness at both code paths
   - Various matrix shapes (square, tall, wide)
2. zig-developer implements dispatcher to make tests GREEN
3. All existing tests must still pass (drop-in replacement)

**Example Results** (Session 481):
- Before: gemm() naive triple-loop → 1.25-2.63 GFLOPS (42-53% of target)
- After: auto-dispatch to gemm_blocked_4x4() → expected 3.5-4.0 GFLOPS (70-80% of target)
- Small matrices: no performance change (threshold preserves naive path)

## SIMD Triangular Solve (trsv_simd) Pattern — Session 497

**When to use**: SIMD-accelerate inner dot product in sequential triangular solve loops.

**Key insight**: Data dependencies prevent outer loop parallelism (x[i] depends on x[j] for j ≠ i), but inner accumulation can be vectorized.

```zig
pub fn trsv_simd(comptime T: type, uplo: u8, trans: u8, diag: u8,
                 A: NDArray(T, 2), x: *NDArray(T, 1)) !void {
    const n = A.shape[0];

    // Allocate temp buffer to preserve original RHS during solve
    const temp = try x.allocator.alloc(T, n);
    defer x.allocator.free(temp);
    @memcpy(temp, x.data);  // temp = copy of b

    const vec_width = comptime simdWidth(T);  // 4 for f64, 8 for f32
    const Vec = @Vector(vec_width, T);

    // 4 cases based on uplo × trans
    if (!is_trans) {
        if (is_upper) {
            // Back substitution: for i = n-1 down to 0
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                var sum: T = temp[i];

                // SIMD main loop for inner dot product
                var j = i + 1;
                while (j + vec_width <= n) : (j += vec_width) {
                    var a_vec: Vec = undefined;
                    var x_vec: Vec = undefined;
                    inline for (0..vec_width) |k| {
                        a_vec[k] = A.data[i * n + j + k];
                        x_vec[k] = x.data[j + k];
                    }
                    const prod = a_vec * x_vec;
                    sum -= @reduce(.Add, prod);  // Accumulate: sum -= Σ(a*x)
                }

                // Scalar tail loop
                while (j < n) : (j += 1) {
                    sum -= A.data[i * n + j] * x.data[j];
                }

                // Solve for x[i]
                x.data[i] = if (!is_unit) sum / A.data[i * n + i] else sum;
            }
        } else {
            // Forward substitution: similar pattern with i in 0..n and j in 0..i
        }
    } else {
        // Transpose cases: similar but access A[j,i] instead of A[i,j]
    }
}
```

**SIMD vectorization breakdown**:
1. **Temporary buffer**: Preserve original RHS (b) because x[i] overwrites as solved
2. **Outer loop**: Sequential (data dependencies on prior x[j])
3. **Inner loop**: Vectorize dot product accumulation
   - Main loop: Process `vec_width` elements at once
   - Use `@Vector(vec_width, T)` and `@reduce(.Add, ...)`
   - SIMD width: 4 for f64, 8 for f32 (256-bit AVX/AVX2)
4. **Tail loop**: Scalar for `n % vec_width` remainder
5. **Unit diagonal**: Skip division when diag='U'

**4 Cases** (uplo × trans):
- **Upper + NoTrans**: Back substitution, `j ∈ [i+1, n)`
- **Lower + NoTrans**: Forward substitution, `j ∈ [0, i)`
- **Upper + Trans**: Forward on A^T, access A[j,i], `j ∈ [0, i)`
- **Lower + Trans**: Back on A^T, access A[j,i], `j ∈ [i+1, n)`

**Testing strategy** (24 tests):
- Basic cases: 2×2 upper/lower × noTrans/trans × unit/nonUnit
- Edge case: 1×1 matrix
- Type support: both f32 and f64
- Large matrices: 64×64 (threshold), 128×128, 256×256
- Error paths: non-square, dimension mismatch
- Memory: no leaks over 10 iterations

**Auto-dispatch integration** (threshold: n >= 64):
```zig
pub fn trsv(comptime T: type, uplo: u8, trans: u8, diag: u8,
            A: NDArray(T, 2), x: *NDArray(T, 1)) !void {
    // ... validation ...
    const n = A.shape[0];

    // Auto-dispatch: use SIMD for large matrices
    if (n >= 64) {
        return try simd_blas.trsv_simd(T, uplo, trans, diag, A, x);
    }

    // Scalar fallback for small matrices (lower overhead)
    // ... scalar implementation ...
}
```

**Performance characteristics**:
- Time: O(n²) sequential (data dependencies prevent parallelism)
- Space: O(n) temporary buffer
- Speedup: 2-4× over scalar on large matrices (vectorized dot product)
- Threshold: 64×64 is empirically optimal (SIMD speedup > overhead at this size)

**Pitfalls to avoid**:
- Don't forget temp buffer copy (original RHS needed for all x[i])
- Don't vectorize outer loop (data dependencies)
- Don't forget tail loop (n % vec_width elements)
- Don't forget unit diagonal special case (skip A[i,i] division)
- Don't forget 4 cases (upper/lower × noTrans/trans)

## FFT Twiddle Factor Caching Pattern — Session 507

**When to use**: Optimize FFT by pre-computing twiddle factors (cos/sin values) instead of computing on-the-fly in butterfly loops.

**Current bottleneck** (src/signal/fft.zig lines 94-96):
```zig
while (j < half_size) : (j += 1) {
    const angle = theta * @as(T, @floatFromInt(j));
    const twiddle = Complex(T).init(@cos(angle), @sin(angle));  // EXPENSIVE
    // ... butterfly operation ...
}
```
- Cost: ~n*log(n)/2 trigonometric evaluations (0.5B trig ops for n=1B FFT)
- Each @cos/@sin: ~100+ CPU cycles, ~10% of total FFT time

**Optimization strategy**:
```zig
pub fn fftCached(comptime T: type, allocator: Allocator, input: []const Complex(T)) ![]Complex(T) {
    const n = input.len;
    if (n == 0) return error.InvalidSize;
    if (!isPowerOfTwo(n)) return error.NotPowerOfTwo;

    // 1. Allocate output and twiddle cache
    var output = try allocator.alloc(Complex(T), n);
    errdefer allocator.free(output);
    @memcpy(output, input);

    bitReversePermutation(Complex(T), output);

    // 2. Pre-compute all twiddle factors for this FFT size
    var twiddles = try allocator.alloc(Complex(T), n);
    defer allocator.free(twiddles);

    // Pre-compute twiddle factors: W_{N}^{k} = exp(-j*2π*k/N) for all k
    // Only need n/2 unique values per stage, but compute all for simplicity
    for (0..n) |k| {
        const angle = -2.0 * math.pi * @as(T, @floatFromInt(k)) / @as(T, @floatFromInt(n));
        twiddles[k] = Complex(T).init(@cos(angle), @sin(angle));
    }

    // 3. Cooley-Tukey FFT using cached twiddles
    var size: usize = 2;
    while (size <= n) : (size *= 2) {
        const half_size = size / 2;
        const twiddle_stride = n / size;  // Index stride in twiddle array

        var k: usize = 0;
        while (k < n) : (k += size) {
            var j: usize = 0;
            while (j < half_size) : (j += 1) {
                const twiddle_idx = j * twiddle_stride;
                const twiddle = twiddles[twiddle_idx];  // LOOKUP instead of compute

                const t = twiddle.mul(output[k + j + half_size]);
                const u = output[k + j];

                output[k + j] = u.add(t);
                output[k + j + half_size] = u.sub(t);
            }
        }
    }

    return output;
}
```

**Key aspects**:
1. **Twiddle indexing**: For each butterfly stage, use `twiddle_stride = n / size` to index correctly
   - Stage 1 (size=2): twiddle_stride = n/2, use twiddles[0], twiddles[n/2], twiddles[n/4], ...
   - Stage 2 (size=4): twiddle_stride = n/4, use twiddles[0], twiddles[n/4], twiddles[n/8], ...
   - General: The `j * twiddle_stride` indexing maps to W_{N}^{j*stride}
2. **Memory tradeoff**: O(n) extra space for twiddles vs O(log n) computation savings
3. **Accuracy**: Should be identical to fft() since we're using same trig computations, just cached
4. **Error paths**: Same as fft() — validate input size and power-of-two

**Test coverage** (17 comprehensive tests):

1. **Correctness (6 tests)**: fftCached() matches fft() exactly
   - Sizes: 8, 16, 32, 256, 512, 4096
   - Tolerances: 1e-9 for small, 1e-7 for large (accumulation error)

2. **Type support (2 tests)**: Both f32 and f64
   - f32 16-point (tolerance 1e-6)
   - f32 256-point (tolerance 1e-5)

3. **Edge cases (2 tests)**:
   - Single point (n=1) — trivial case
   - Non-power-of-two error handling

4. **Numerical properties (4 tests)**:
   - Impulse response at different positions
   - Real sine wave (validates frequency peak)
   - Complex exponential (validates complex handling)
   - DC constant signal (validates spectral shape)

5. **Mathematical invariants (1 test)**:
   - Parseval's theorem: sum(|FFT|²) = N * sum(|input|²)

6. **Memory safety (1 test)**:
   - 10 iterations with testing.allocator (leak detection)

7. **Error handling (1 test)**:
   - Empty input validation

**Performance expectations**:
- Time: O(n log n) same as fft(), but with smaller constant factor
- Space: O(n) for output + O(n) for twiddle cache
- Speedup: 2-3× for large FFTs (n >= 256) due to eliminated trig ops
- Latency: Slightly higher for small FFTs (n < 64) due to pre-compute overhead
- Microarchitecture: Table lookups → better cache locality than repeated trig computation

**Pitfalls to avoid**:
- Don't forget twiddle_stride = n / size for each butterfly stage
- Don't mix up twiddle indexing between stages (stride changes!)
- Don't assume identical floating-point results (minor rounding differences OK within 1e-6)
- Don't forget to defer/free twiddle allocation
- Don't pre-compute all exp(-j*2π*k/N) naively in the butterfly loop (defeats purpose)

## Heavy-Tailed Distribution Testing Pattern (Landau, Session 726+)

For distributions with non-closed-form PDFs (Schorr approximation, Spence-Dirichlet, etc.):

**Test categories**:

1. **Parameter Validation (6 tests)**:
   - Valid init with various param combinations
   - Invalid scale (c ≤ 0) — boundary and strictly negative
   - NaN/Inf for location and scale parameters
   - f32 type support

2. **PDF Properties (6 tests)**:
   - Positive everywhere on support (check at multiple x)
   - Tail behavior: decay to ~0 far negative, monotone decrease positive
   - Reference values from tables (CERN/ROOT): pdf(-1), pdf(0), pdf(1) within 1e-3
   - logpdf = log(pdf) for finite positive pdf
   - logpdf finite everywhere

3. **CDF Properties (7 tests)**:
   - Monotone increasing (test 7+ points)
   - CDF limits: F(-100) ≈ 0, F(+100) ≈ 1
   - CDF + SF = 1 at multiple points (tolerance 1e-10)
   - Known quantile: median ≈ 0.765 (within 0.01 for standard Landau)

4. **Quantile (Inverse CDF) (8 tests)**:
   - Monotone increasing in p
   - Roundtrip: cdf(quantile(p)) ≈ p for p in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
   - Error for p < 0, p > 1, NaN → error.InvalidProbability
   - Tolerance: 1e-4 for roundtrip (numerical inversion via bisection)

5. **Mode (Peak) Properties (2 tests)**:
   - Mode ≈ -0.222 for standard Landau (within 0.01)
   - Mode is finite; PDF at mode > PDF at nearby points

6. **Scale Properties (3 tests)**:
   - Location shift: pdf(x; μ, c) = (1/c) * pdf((x-μ)/c; 0, 1)
   - Scale invariance: pdf(x; μ, c) scales as 1/c
   - CDF location shift: F(x; μ, c) = F((x-μ)/c; 0, 1)

7. **Moments (3 tests)**:
   - Mean/variance/entropy all finite and positive
   - Larger scale → larger variance
   - Location shift doesn't change variance

8. **Sampling (2 tests)**:
   - Sample returns finite values (100+ iterations)
   - Empirical mean ≈ analytical mean (tolerance 0.5 for 1000 samples)

9. **Supplementary CDF Checks (2 tests)**:
   - CDF reaches 0.25 before median
   - CDF reaches 0.75 after median
   - SF at median ≈ 0.5

**Key numerical facts for standard Landau (μ=0, c=1)**:
- Mode: x_m ≈ -0.2224
- Median: x_0.5 ≈ 0.7650
- PDF(-1.0) ≈ 0.1791
- PDF(0.0) ≈ 0.1800
- PDF(1.0) ≈ 0.0848
- Mean: does not exist (heavy tail)
- Variance: does not exist
- Support: (-∞, +∞)

**Implementation hints**:
- Use Schorr approximation (polynomial fit + numerical methods)
- CDF via numerical integration or asymptotic expansions (fit to tables)
- Quantile via bisection (O(log(1/ε)) iterations, e.g., 64 iterations)
- Mode via bisection on score condition (d log pdf / dx = 0)
- Mean/variance/entropy via 500-point numerical quadrature
- Sample via inverse-CDF transform

**Tolerance guidelines**:
- PDF exact values: 1e-3 (table-based approximation error)
- CDF/SF: 1e-4 (numerical integration error)
- Quantile roundtrip: 1e-4 (bisection error)
- Moments (mean/var): 0.5 (empirical estimate, 1000 samples)

**Pitfalls**:
- Don't assume mean/variance exist — Landau has infinite tails
- Don't use direct FFT for CDF inversion — use bisection or Spence expansion
- Don't hardcode reference values in expected calculations — use actual computed values
- Don't forget mode can be found via bisection, not closed form
- Don't assume CDF follows Normal-like S-curve — Landau is skewed left (peak negative)
