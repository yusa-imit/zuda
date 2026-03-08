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
