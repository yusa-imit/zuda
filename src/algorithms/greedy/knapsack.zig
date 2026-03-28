const std = @import("std");
const testing = std.testing;

/// Item with value and weight
pub const Item = struct {
    value: f64,
    weight: f64,
    id: usize,

    pub fn valuePerWeight(self: Item) f64 {
        return if (self.weight > 0) self.value / self.weight else 0;
    }
};

/// Solves fractional knapsack problem using greedy approach
///
/// Algorithm: Sort by value-to-weight ratio, greedily fill knapsack
///
/// Time: O(n log n) — sorting dominates
/// Space: O(n) — sorted items
///
/// Returns: Total value achieved
///
/// Example:
/// ```zig
/// const items = [_]Item{
///     .{ .value = 60, .weight = 10, .id = 0 },
///     .{ .value = 100, .weight = 20, .id = 1 },
///     .{ .value = 120, .weight = 30, .id = 2 },
/// };
/// const value = fractionalKnapsack(&items, 50);
/// // value = 240 (all of item 1, all of item 0, 2/3 of item 2)
/// ```
pub fn fractionalKnapsack(
    items: []const Item,
    capacity: f64,
) f64 {
    if (items.len == 0 or capacity <= 0) return 0;

    // Sort by value-to-weight ratio (descending)
    const sorted = std.heap.page_allocator.alloc(Item, items.len) catch return 0;
    defer std.heap.page_allocator.free(sorted);
    @memcpy(sorted, items);

    std.mem.sort(Item, sorted, {}, greaterByRatio);

    var total_value: f64 = 0;
    var remaining_capacity = capacity;

    for (sorted) |item| {
        if (remaining_capacity <= 0) break;

        if (item.weight <= remaining_capacity) {
            // Take entire item
            total_value += item.value;
            remaining_capacity -= item.weight;
        } else {
            // Take fraction of item
            const fraction = remaining_capacity / item.weight;
            total_value += item.value * fraction;
            remaining_capacity = 0;
        }
    }

    return total_value;
}

/// Solves fractional knapsack and returns selected items with fractions
///
/// Time: O(n log n)
/// Space: O(n)
pub fn fractionalKnapsackDetailed(
    allocator: std.mem.Allocator,
    items: []const Item,
    capacity: f64,
) !std.ArrayList(ItemSelection) {
    var result = std.ArrayList(ItemSelection).init(allocator);
    errdefer result.deinit();

    if (items.len == 0 or capacity <= 0) return result;

    const sorted = try allocator.alloc(Item, items.len);
    defer allocator.free(sorted);
    @memcpy(sorted, items);

    std.mem.sort(Item, sorted, {}, greaterByRatio);

    var remaining_capacity = capacity;

    for (sorted) |item| {
        if (remaining_capacity <= 0) break;

        if (item.weight <= remaining_capacity) {
            try result.append(.{
                .item_id = item.id,
                .fraction = 1.0,
                .value_taken = item.value,
            });
            remaining_capacity -= item.weight;
        } else {
            const fraction = remaining_capacity / item.weight;
            try result.append(.{
                .item_id = item.id,
                .fraction = fraction,
                .value_taken = item.value * fraction,
            });
            remaining_capacity = 0;
        }
    }

    return result;
}

pub const ItemSelection = struct {
    item_id: usize,
    fraction: f64, // 0.0 to 1.0
    value_taken: f64,
};

fn greaterByRatio(_: void, a: Item, b: Item) bool {
    return a.valuePerWeight() > b.valuePerWeight();
}

/// 0/1 Knapsack using greedy approximation (not optimal, but fast)
///
/// Algorithm: Sort by value-to-weight ratio, take whole items until full
///
/// Time: O(n log n)
/// Space: O(n)
///
/// Note: This is an approximation. For exact solution, use dynamic programming.
pub fn zeroOneKnapsackGreedy(
    allocator: std.mem.Allocator,
    items: []const Item,
    capacity: f64,
) !std.ArrayList(usize) {
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    if (items.len == 0 or capacity <= 0) return result;

    const sorted = try allocator.alloc(Item, items.len);
    defer allocator.free(sorted);
    @memcpy(sorted, items);

    std.mem.sort(Item, sorted, {}, greaterByRatio);

    var remaining_capacity = capacity;

    for (sorted) |item| {
        if (item.weight <= remaining_capacity) {
            try result.append(item.id);
            remaining_capacity -= item.weight;
        }
    }

    return result;
}

// Tests
test "fractional knapsack - basic case" {
    const items = [_]Item{
        .{ .value = 60, .weight = 10, .id = 0 },
        .{ .value = 100, .weight = 20, .id = 1 },
        .{ .value = 120, .weight = 30, .id = 2 },
    };
    const capacity = 50.0;

    const value = fractionalKnapsack(&items, capacity);

    // Expected: item 1 (100), item 0 (60), 2/3 of item 2 (80) = 240
    try testing.expectApproxEqAbs(@as(f64, 240), value, 0.01);
}

test "fractional knapsack - detailed selection" {
    const items = [_]Item{
        .{ .value = 60, .weight = 10, .id = 0 },
        .{ .value = 100, .weight = 20, .id = 1 },
        .{ .value = 120, .weight = 30, .id = 2 },
    };
    const capacity = 50.0;

    var selections = try fractionalKnapsackDetailed(testing.allocator, &items, capacity);
    defer selections.deinit();

    var total_value: f64 = 0;
    for (selections.items) |sel| {
        total_value += sel.value_taken;
    }

    try testing.expectApproxEqAbs(@as(f64, 240), total_value, 0.01);
    try testing.expectEqual(@as(usize, 3), selections.items.len);
}

test "fractional knapsack - zero capacity" {
    const items = [_]Item{
        .{ .value = 60, .weight = 10, .id = 0 },
    };
    const value = fractionalKnapsack(&items, 0);
    try testing.expectEqual(@as(f64, 0), value);
}

test "fractional knapsack - empty items" {
    const items: []const Item = &.{};
    const value = fractionalKnapsack(items, 100);
    try testing.expectEqual(@as(f64, 0), value);
}

test "fractional knapsack - large capacity" {
    const items = [_]Item{
        .{ .value = 60, .weight = 10, .id = 0 },
        .{ .value = 100, .weight = 20, .id = 1 },
    };
    const capacity = 1000.0;

    const value = fractionalKnapsack(&items, capacity);
    // Should take all items
    try testing.expectEqual(@as(f64, 160), value);
}

test "fractional knapsack - single item fits" {
    const items = [_]Item{
        .{ .value = 100, .weight = 50, .id = 0 },
    };
    const capacity = 100.0;

    const value = fractionalKnapsack(&items, capacity);
    try testing.expectEqual(@as(f64, 100), value);
}

test "fractional knapsack - single item partial" {
    const items = [_]Item{
        .{ .value = 100, .weight = 50, .id = 0 },
    };
    const capacity = 25.0;

    const value = fractionalKnapsack(&items, capacity);
    try testing.expectEqual(@as(f64, 50), value); // Half the value
}

test "0/1 knapsack greedy - basic case" {
    const items = [_]Item{
        .{ .value = 60, .weight = 10, .id = 0 },
        .{ .value = 100, .weight = 20, .id = 1 },
        .{ .value = 120, .weight = 30, .id = 2 },
    };
    const capacity = 50.0;

    var selected = try zeroOneKnapsackGreedy(testing.allocator, &items, capacity);
    defer selected.deinit();

    // Greedy selects items 1 and 0 (total weight 30, value 160)
    try testing.expect(selected.items.len >= 2);

    var total_value: f64 = 0;
    var total_weight: f64 = 0;
    for (selected.items) |id| {
        total_value += items[id].value;
        total_weight += items[id].weight;
    }

    try testing.expect(total_weight <= capacity);
    try testing.expect(total_value >= 160);
}

test "0/1 knapsack greedy - all items fit" {
    const items = [_]Item{
        .{ .value = 10, .weight = 5, .id = 0 },
        .{ .value = 20, .weight = 10, .id = 1 },
    };
    const capacity = 100.0;

    var selected = try zeroOneKnapsackGreedy(testing.allocator, &items, capacity);
    defer selected.deinit();

    try testing.expectEqual(@as(usize, 2), selected.items.len);
}
