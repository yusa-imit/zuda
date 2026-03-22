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

