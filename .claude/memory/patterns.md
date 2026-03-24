# zuda Code Patterns

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

