const std = @import("std");
const Allocator = std.mem.Allocator;

/// Knapsack Problem Algorithms
///
/// Provides solutions for various knapsack problem variants.

pub const Item = struct {
    weight: usize,
    value: usize,

    pub fn init(weight: usize, value: usize) Item {
        return .{ .weight = weight, .value = value };
    }
};

/// 0/1 Knapsack: Each item can be taken at most once
/// Time: O(n*W) | Space: O(n*W) where W is capacity
pub fn zeroOne(items: []const Item, capacity: usize) !usize {
    if (items.len == 0 or capacity == 0) return 0;

    const allocator = std.heap.page_allocator;
    const n = items.len;

    // dp[i][w] = max value using first i items with capacity w
    const dp = try allocator.alloc([]usize, n + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..n + 1) |i| {
        dp[i] = try allocator.alloc(usize, capacity + 1);
        @memset(dp[i], 0);
    }

    for (1..n + 1) |i| {
        const item = items[i - 1];
        for (0..capacity + 1) |w| {
            // Don't take item i
            dp[i][w] = dp[i - 1][w];

            // Take item i if it fits
            if (item.weight <= w) {
                const value_with = dp[i - 1][w - item.weight] + item.value;
                dp[i][w] = @max(dp[i][w], value_with);
            }
        }
    }

    return dp[n][capacity];
}

/// 0/1 Knapsack with space optimization
/// Time: O(n*W) | Space: O(W)
pub fn zeroOneOptimized(items: []const Item, capacity: usize) !usize {
    if (items.len == 0 or capacity == 0) return 0;

    const allocator = std.heap.page_allocator;
    const dp = try allocator.alloc(usize, capacity + 1);
    defer allocator.free(dp);

    @memset(dp, 0);

    for (items) |item| {
        // Traverse backwards to avoid using the same item multiple times
        var w = capacity;
        while (w >= item.weight) : (w -= 1) {
            dp[w] = @max(dp[w], dp[w - item.weight] + item.value);
        }
    }

    return dp[capacity];
}

/// 0/1 Knapsack with item selection tracking
/// Time: O(n*W) | Space: O(n*W)
pub fn zeroOneWithItems(allocator: Allocator, items: []const Item, capacity: usize) !struct {
    value: usize,
    selected: []bool,
} {
    if (items.len == 0 or capacity == 0) {
        const selected = try allocator.alloc(bool, items.len);
        @memset(selected, false);
        return .{ .value = 0, .selected = selected };
    }

    const n = items.len;

    const dp = try allocator.alloc([]usize, n + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..n + 1) |i| {
        dp[i] = try allocator.alloc(usize, capacity + 1);
        @memset(dp[i], 0);
    }

    for (1..n + 1) |i| {
        const item = items[i - 1];
        for (0..capacity + 1) |w| {
            dp[i][w] = dp[i - 1][w];
            if (item.weight <= w) {
                const value_with = dp[i - 1][w - item.weight] + item.value;
                dp[i][w] = @max(dp[i][w], value_with);
            }
        }
    }

    // Backtrack to find selected items
    const selected = try allocator.alloc(bool, n);
    @memset(selected, false);

    var i = n;
    var w = capacity;

    while (i > 0 and w > 0) {
        if (dp[i][w] != dp[i - 1][w]) {
            selected[i - 1] = true;
            w -= items[i - 1].weight;
        }
        i -= 1;
    }

    return .{ .value = dp[n][capacity], .selected = selected };
}

/// Unbounded Knapsack: Each item can be taken unlimited times
/// Time: O(n*W) | Space: O(W)
pub fn unbounded(items: []const Item, capacity: usize) !usize {
    if (items.len == 0 or capacity == 0) return 0;

    const allocator = std.heap.page_allocator;
    const dp = try allocator.alloc(usize, capacity + 1);
    defer allocator.free(dp);

    @memset(dp, 0);

    for (1..capacity + 1) |w| {
        for (items) |item| {
            if (item.weight <= w) {
                dp[w] = @max(dp[w], dp[w - item.weight] + item.value);
            }
        }
    }

    return dp[capacity];
}

/// Unbounded Knapsack with item count tracking
/// Time: O(n*W) | Space: O(W)
pub fn unboundedWithCounts(allocator: Allocator, items: []const Item, capacity: usize) !struct {
    value: usize,
    counts: []usize,
} {
    if (items.len == 0 or capacity == 0) {
        const counts = try allocator.alloc(usize, items.len);
        @memset(counts, 0);
        return .{ .value = 0, .counts = counts };
    }

    const n = items.len;

    const dp = try allocator.alloc(usize, capacity + 1);
    defer allocator.free(dp);
    @memset(dp, 0);

    const parent = try allocator.alloc(?struct { item_idx: usize, prev_w: usize }, capacity + 1);
    defer allocator.free(parent);
    @memset(parent, null);

    for (1..capacity + 1) |w| {
        for (items, 0..) |item, idx| {
            if (item.weight <= w) {
                const new_value = dp[w - item.weight] + item.value;
                if (new_value > dp[w]) {
                    dp[w] = new_value;
                    parent[w] = .{ .item_idx = idx, .prev_w = w - item.weight };
                }
            }
        }
    }

    // Backtrack to count items
    const counts = try allocator.alloc(usize, n);
    @memset(counts, 0);

    var w = capacity;
    while (parent[w]) |p| {
        counts[p.item_idx] += 1;
        w = p.prev_w;
    }

    return .{ .value = dp[capacity], .counts = counts };
}

/// Fractional Knapsack: Items can be taken partially (greedy solution)
/// Time: O(n log n) | Space: O(n)
pub fn fractional(allocator: Allocator, items: []const Item, capacity: usize) !f64 {
    if (items.len == 0 or capacity == 0) return 0.0;

    // Create sorted copy by value/weight ratio
    const IndexedItem = struct {
        item: Item,
        index: usize,
        ratio: f64,

        pub fn lessThan(_: void, a: @This(), b: @This()) bool {
            return a.ratio > b.ratio; // Descending order
        }
    };

    const indexed = try allocator.alloc(IndexedItem, items.len);
    defer allocator.free(indexed);

    for (items, 0..) |item, i| {
        indexed[i] = .{
            .item = item,
            .index = i,
            .ratio = @as(f64, @floatFromInt(item.value)) / @as(f64, @floatFromInt(item.weight)),
        };
    }

    std.mem.sort(IndexedItem, indexed, {}, IndexedItem.lessThan);

    var total_value: f64 = 0.0;
    var remaining_capacity = capacity;

    for (indexed) |indexed_item| {
        const item = indexed_item.item;
        if (remaining_capacity == 0) break;

        if (item.weight <= remaining_capacity) {
            // Take whole item
            total_value += @floatFromInt(item.value);
            remaining_capacity -= item.weight;
        } else {
            // Take fraction of item
            const fraction = @as(f64, @floatFromInt(remaining_capacity)) / @as(f64, @floatFromInt(item.weight));
            total_value += @as(f64, @floatFromInt(item.value)) * fraction;
            remaining_capacity = 0;
        }
    }

    return total_value;
}

/// Bounded Knapsack: Each item can be taken up to a limited count
/// Time: O(n*W*K) where K is max count | Space: O(W)
pub fn bounded(items: []const Item, counts: []const usize, capacity: usize) !usize {
    if (items.len == 0 or capacity == 0) return 0;
    std.debug.assert(items.len == counts.len);

    const allocator = std.heap.page_allocator;
    const dp = try allocator.alloc(usize, capacity + 1);
    defer allocator.free(dp);

    @memset(dp, 0);

    for (items, counts) |item, count| {
        // Traverse backwards for each item
        var w = capacity;
        while (w >= item.weight) : (w -= 1) {
            // Try taking 1, 2, ..., count copies of this item
            var k: usize = 1;
            while (k <= count and k * item.weight <= w) : (k += 1) {
                dp[w] = @max(dp[w], dp[w - k * item.weight] + k * item.value);
            }
        }
    }

    return dp[capacity];
}

// ============================================================================
// Tests
// ============================================================================

test "Knapsack: 0/1 empty" {
    const items: []const Item = &[_]Item{};
    try std.testing.expectEqual(0, try zeroOne(items, 10));
    try std.testing.expectEqual(0, try zeroOneOptimized(items, 10));

    const result = try zeroOneWithItems(std.testing.allocator, items, 10);
    defer std.testing.allocator.free(result.selected);
    try std.testing.expectEqual(0, result.value);
}

test "Knapsack: 0/1 zero capacity" {
    const items = [_]Item{
        Item.init(2, 3),
        Item.init(3, 4),
    };

    try std.testing.expectEqual(0, try zeroOne(&items, 0));
    try std.testing.expectEqual(0, try zeroOneOptimized(&items, 0));
}

test "Knapsack: 0/1 classic example" {
    const items = [_]Item{
        Item.init(2, 3), // value/weight = 1.5
        Item.init(3, 4), // value/weight = 1.33
        Item.init(4, 5), // value/weight = 1.25
        Item.init(5, 6), // value/weight = 1.2
    };
    const capacity = 8;

    // Optimal: items 0 and 3 (weight 7, value 9)
    try std.testing.expectEqual(9, try zeroOne(&items, capacity));
    try std.testing.expectEqual(9, try zeroOneOptimized(&items, capacity));

    const result = try zeroOneWithItems(std.testing.allocator, &items, capacity);
    defer std.testing.allocator.free(result.selected);
    try std.testing.expectEqual(9, result.value);
    try std.testing.expect(result.selected[0] or result.selected[1] or result.selected[2] or result.selected[3]);
}

test "Knapsack: 0/1 all items fit" {
    const items = [_]Item{
        Item.init(1, 2),
        Item.init(2, 3),
        Item.init(3, 4),
    };
    const capacity = 10;

    // All items fit: total value = 9
    try std.testing.expectEqual(9, try zeroOne(&items, capacity));
    try std.testing.expectEqual(9, try zeroOneOptimized(&items, capacity));
}

test "Knapsack: 0/1 single item" {
    const items = [_]Item{Item.init(5, 10)};

    try std.testing.expectEqual(10, try zeroOne(&items, 10));
    try std.testing.expectEqual(0, try zeroOne(&items, 4)); // Doesn't fit
}

test "Knapsack: unbounded example" {
    const items = [_]Item{
        Item.init(2, 3),
        Item.init(3, 4),
        Item.init(4, 5),
    };
    const capacity = 8;

    // Unbounded: take item 0 four times (weight 8, value 12)
    const value = try unbounded(&items, capacity);
    try std.testing.expect(value >= 12);
}

test "Knapsack: unbounded with counts" {
    const items = [_]Item{
        Item.init(2, 3),
        Item.init(3, 4),
    };
    const capacity = 6;

    const result = try unboundedWithCounts(std.testing.allocator, &items, capacity);
    defer std.testing.allocator.free(result.counts);

    // Should take item 0 three times (value 9) or item 1 twice (value 8)
    try std.testing.expect(result.value >= 8);
}

test "Knapsack: fractional" {
    const items = [_]Item{
        Item.init(10, 60), // ratio 6.0
        Item.init(20, 100), // ratio 5.0
        Item.init(30, 120), // ratio 4.0
    };
    const capacity = 50;

    // Take all of item 0 (10kg, 60), all of item 1 (20kg, 100),
    // and 20kg of item 2 (20/30 * 120 = 80)
    // Total: 240
    const value = try fractional(std.testing.allocator, &items, capacity);
    try std.testing.expectApproxEqAbs(240.0, value, 0.01);
}

test "Knapsack: fractional vs 0/1 difference" {
    const items = [_]Item{
        Item.init(10, 60),
        Item.init(20, 100),
        Item.init(30, 120),
    };
    const capacity = 50;

    const frac = try fractional(std.testing.allocator, &items, capacity);
    const int = try zeroOneOptimized(&items, capacity);

    // Fractional should be >= integer solution
    try std.testing.expect(frac >= @as(f64, @floatFromInt(int)));
}

test "Knapsack: bounded" {
    const items = [_]Item{
        Item.init(2, 3),
        Item.init(3, 4),
    };
    const counts = [_]usize{ 2, 1 }; // Can take item 0 at most twice, item 1 once
    const capacity = 7;

    // Optimal: 2x item 0 + 1x item 1 = weight 7, value 10
    try std.testing.expectEqual(10, try bounded(&items, &counts, capacity));
}

test "Knapsack: 0/1 large example" {
    const allocator = std.testing.allocator;
    const n = 100;
    const items = try allocator.alloc(Item, n);
    defer allocator.free(items);

    for (0..n) |i| {
        items[i] = Item.init(i + 1, (i + 1) * 2);
    }

    const capacity = 200;
    const value = try zeroOneOptimized(items, capacity);
    try std.testing.expect(value > 0);
}

test "Knapsack: space optimization correctness" {
    const items = [_]Item{
        Item.init(1, 1),
        Item.init(3, 4),
        Item.init(4, 5),
        Item.init(5, 7),
    };
    const capacity = 7;

    const v1 = try zeroOne(&items, capacity);
    const v2 = try zeroOneOptimized(&items, capacity);

    try std.testing.expectEqual(v1, v2);
}

test "Knapsack: identical items 0/1" {
    const items = [_]Item{
        Item.init(3, 5),
        Item.init(3, 5),
        Item.init(3, 5),
    };
    const capacity = 6;

    // Can only take 2 items (0/1 constraint)
    try std.testing.expectEqual(10, try zeroOneOptimized(&items, capacity));
}

test "Knapsack: identical items unbounded" {
    const items = [_]Item{
        Item.init(3, 5),
    };
    const capacity = 9;

    // Can take 3 items
    try std.testing.expectEqual(15, try unbounded(&items, capacity));
}

test "Knapsack: selection tracking" {
    const items = [_]Item{
        Item.init(5, 10),
        Item.init(4, 8),
        Item.init(6, 12),
        Item.init(3, 6),
    };
    const capacity = 10;

    const result = try zeroOneWithItems(std.testing.allocator, &items, capacity);
    defer std.testing.allocator.free(result.selected);

    // Verify selected items don't exceed capacity
    var total_weight: usize = 0;
    var total_value: usize = 0;
    for (result.selected, items) |selected, item| {
        if (selected) {
            total_weight += item.weight;
            total_value += item.value;
        }
    }

    try std.testing.expect(total_weight <= capacity);
    try std.testing.expectEqual(result.value, total_value);
}
