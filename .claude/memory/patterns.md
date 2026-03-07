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
