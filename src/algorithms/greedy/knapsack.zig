//! Greedy knapsack: the optimal fractional knapsack and a 0/1 approximation, both by descending
//! value-to-weight ratio.
//!
//! Items with a non-positive value are never selected, and non-positive capacity selects
//! nothing. An item with zero weight and positive value is free, so it ranks first.
//!
//! Ownership: the sorted copy of the items is freed before every function returns. The
//! `ArrayList` results are owned by the caller, who releases them with `deinit(gpa)` using the
//! same allocator.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

/// Item with value and weight
pub const Item = struct {
    value: f64,
    weight: f64,
    id: usize,

    /// Value per unit of weight. A weightless item with positive value is infinitely good, and
    /// anything else without weight is worth nothing.
    pub fn valuePerWeight(self: Item) f64 {
        if (self.weight > 0) return self.value / self.weight;
        if (self.value > 0) return std.math.inf(f64);
        return 0;
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
/// const value = try fractionalKnapsack(gpa, &items, 50);
/// // value = 240 (all of item 1, all of item 0, 2/3 of item 2)
/// ```
pub fn fractionalKnapsack(
    gpa: Allocator,
    items: []const Item,
    capacity: f64,
) !f64 {
    if (items.len == 0 or capacity <= 0) return 0;

    const sorted = try sortedByRatio(gpa, items);
    defer gpa.free(sorted);

    var total_value: f64 = 0;
    var remaining_capacity = capacity;

    for (sorted) |item| {
        if (remaining_capacity <= 0) break;
        if (item.value <= 0) continue;

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

    assert(remaining_capacity <= capacity);
    return total_value;
}

/// Solves fractional knapsack and returns selected items with fractions
///
/// Time: O(n log n)
/// Space: O(n)
pub fn fractionalKnapsackDetailed(
    gpa: Allocator,
    items: []const Item,
    capacity: f64,
) !ArrayList(ItemSelection) {
    var result: ArrayList(ItemSelection) = .empty;
    errdefer result.deinit(gpa);

    if (items.len == 0 or capacity <= 0) return result;

    const sorted = try sortedByRatio(gpa, items);
    defer gpa.free(sorted);

    try result.ensureTotalCapacityPrecise(gpa, items.len);

    var remaining_capacity = capacity;

    for (sorted) |item| {
        if (remaining_capacity <= 0) break;
        if (item.value <= 0) continue;

        if (item.weight <= remaining_capacity) {
            result.appendAssumeCapacity(.{
                .item_id = item.id,
                .fraction = 1.0,
                .value_taken = item.value,
            });
            remaining_capacity -= item.weight;
        } else {
            const fraction = remaining_capacity / item.weight;
            result.appendAssumeCapacity(.{
                .item_id = item.id,
                .fraction = fraction,
                .value_taken = item.value * fraction,
            });
            remaining_capacity = 0;
        }
    }

    assert(remaining_capacity <= capacity);
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

/// Returns a freshly allocated copy of `items` sorted by descending ratio; the caller frees it.
fn sortedByRatio(gpa: Allocator, items: []const Item) ![]Item {
    assert(items.len > 0);

    const sorted = try gpa.alloc(Item, items.len);
    errdefer gpa.free(sorted);

    @memcpy(sorted, items);
    std.mem.sort(Item, sorted, {}, greaterByRatio);
    assert(sorted.len == items.len);
    return sorted;
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
    gpa: Allocator,
    items: []const Item,
    capacity: f64,
) !ArrayList(usize) {
    var result: ArrayList(usize) = .empty;
    errdefer result.deinit(gpa);

    if (items.len == 0 or capacity <= 0) return result;

    const sorted = try sortedByRatio(gpa, items);
    defer gpa.free(sorted);

    try result.ensureTotalCapacityPrecise(gpa, items.len);

    var remaining_capacity = capacity;

    for (sorted) |item| {
        if (item.value <= 0) continue;

        if (item.weight <= remaining_capacity) {
            result.appendAssumeCapacity(item.id);
            remaining_capacity -= item.weight;
        }
    }

    assert(remaining_capacity <= capacity);
    return result;
}

test "fractional knapsack - basic case" {
    const items = [_]Item{
        .{ .value = 60, .weight = 10, .id = 0 },
        .{ .value = 100, .weight = 20, .id = 1 },
        .{ .value = 120, .weight = 30, .id = 2 },
    };
    const capacity = 50.0;

    const value = try fractionalKnapsack(testing.allocator, &items, capacity);

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
    defer selections.deinit(testing.allocator);

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
    const value = try fractionalKnapsack(testing.allocator, &items, 0);
    try testing.expectEqual(@as(f64, 0), value);
}

test "fractional knapsack - empty items" {
    const items: []const Item = &.{};
    const value = try fractionalKnapsack(testing.allocator, items, 100);
    try testing.expectEqual(@as(f64, 0), value);
}

test "fractional knapsack - large capacity" {
    const items = [_]Item{
        .{ .value = 60, .weight = 10, .id = 0 },
        .{ .value = 100, .weight = 20, .id = 1 },
    };
    const capacity = 1000.0;

    const value = try fractionalKnapsack(testing.allocator, &items, capacity);
    // Should take all items
    try testing.expectEqual(@as(f64, 160), value);
}

test "fractional knapsack - single item fits" {
    const items = [_]Item{
        .{ .value = 100, .weight = 50, .id = 0 },
    };
    const capacity = 100.0;

    const value = try fractionalKnapsack(testing.allocator, &items, capacity);
    try testing.expectEqual(@as(f64, 100), value);
}

test "fractional knapsack - single item partial" {
    const items = [_]Item{
        .{ .value = 100, .weight = 50, .id = 0 },
    };
    const capacity = 25.0;

    const value = try fractionalKnapsack(testing.allocator, &items, capacity);
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
    defer selected.deinit(testing.allocator);

    // Greedy selects items 0 and 1 (total weight 30, value 160); item 2 no longer fits.
    try testing.expectEqualSlices(usize, &.{ 0, 1 }, selected.items);

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
    defer selected.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), selected.items.len);
}

test "fractional knapsack - capacity exactly fits all items" {
    const items = [_]Item{
        .{ .value = 10, .weight = 4, .id = 0 },
        .{ .value = 20, .weight = 6, .id = 1 },
        .{ .value = 15, .weight = 5, .id = 2 },
    };
    const capacity = 15.0;

    const value = try fractionalKnapsack(testing.allocator, &items, capacity);
    try testing.expectApproxEqAbs(@as(f64, 45.0), value, 1e-9);
}

test "fractional knapsack - capacity larger than all items" {
    const items = [_]Item{
        .{ .value = 5, .weight = 1, .id = 0 },
        .{ .value = 3, .weight = 2, .id = 1 },
    };
    const capacity = 100.0;

    const value = try fractionalKnapsack(testing.allocator, &items, capacity);
    try testing.expectApproxEqAbs(@as(f64, 8.0), value, 1e-9);
}

test "0/1 knapsack greedy - greedy suboptimal known case" {
    const items = [_]Item{
        .{ .value = 10, .weight = 6, .id = 0 },
        .{ .value = 7, .weight = 4, .id = 1 },
        .{ .value = 7, .weight = 4, .id = 2 },
    };
    const capacity = 8.0;

    var selected = try zeroOneKnapsackGreedy(testing.allocator, &items, capacity);
    defer selected.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), selected.items.len);

    var total_value: f64 = 0;
    for (selected.items) |id| {
        total_value += items[id].value;
    }
    try testing.expectApproxEqAbs(@as(f64, 14.0), total_value, 1e-9);
}

test "fractional knapsack - detailed selection fractions sum to one item" {
    const items = [_]Item{
        .{ .value = 100, .weight = 50, .id = 0 },
        .{ .value = 60, .weight = 20, .id = 1 },
    };
    const capacity = 30.0;

    var selections = try fractionalKnapsackDetailed(testing.allocator, &items, capacity);
    defer selections.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), selections.items.len);

    var item_id_1_found = false;
    var item_id_0_found = false;

    for (selections.items) |sel| {
        if (sel.item_id == 1) {
            try testing.expectApproxEqAbs(@as(f64, 1.0), sel.fraction, 1e-9);
            item_id_1_found = true;
        } else if (sel.item_id == 0) {
            try testing.expectApproxEqAbs(@as(f64, 0.2), sel.fraction, 1e-9);
            item_id_0_found = true;
        }
    }

    try testing.expect(item_id_0_found);
    try testing.expect(item_id_1_found);

    var total_value: f64 = 0;
    for (selections.items) |sel| {
        total_value += sel.value_taken;
    }
    try testing.expectApproxEqAbs(@as(f64, 80.0), total_value, 1e-9);
}

test "fractional knapsack - memory safety loop" {
    const items = [_]Item{
        .{ .value = 10, .weight = 5, .id = 0 },
        .{ .value = 20, .weight = 10, .id = 1 },
    };

    for (0..10) |_| {
        const value = try fractionalKnapsack(testing.allocator, &items, 12.0);
        try testing.expect(value > 0);

        var selected = try zeroOneKnapsackGreedy(testing.allocator, &items, 12.0);
        defer selected.deinit(testing.allocator);
        try testing.expect(selected.items.len > 0);
    }
}

test "fractional knapsack - free items are taken even when the sack is otherwise full" {
    const items = [_]Item{
        .{ .value = 60, .weight = 10, .id = 0 },
        .{ .value = 5, .weight = 0, .id = 1 },
    };

    const value = try fractionalKnapsack(testing.allocator, &items, 10);
    try testing.expectApproxEqAbs(@as(f64, 65), value, 1e-9);

    var selected = try zeroOneKnapsackGreedy(testing.allocator, &items, 10);
    defer selected.deinit(testing.allocator);

    try testing.expectEqualSlices(usize, &.{ 1, 0 }, selected.items);
}

test "knapsack - items without positive value are never selected" {
    const items = [_]Item{
        .{ .value = -10, .weight = 1, .id = 0 },
        .{ .value = 0, .weight = 1, .id = 1 },
        .{ .value = 8, .weight = 2, .id = 2 },
    };

    const value = try fractionalKnapsack(testing.allocator, &items, 100);
    try testing.expectApproxEqAbs(@as(f64, 8), value, 1e-9);

    var selections = try fractionalKnapsackDetailed(testing.allocator, &items, 100);
    defer selections.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), selections.items.len);
    try testing.expectEqual(@as(usize, 2), selections.items[0].item_id);
}

test "knapsack - no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const items = [_]Item{
                .{ .value = 60, .weight = 10, .id = 0 },
                .{ .value = 100, .weight = 20, .id = 1 },
                .{ .value = 120, .weight = 30, .id = 2 },
            };
            const value = try fractionalKnapsack(gpa, &items, 50);
            try testing.expectApproxEqAbs(@as(f64, 240), value, 1e-9);

            var selections = try fractionalKnapsackDetailed(gpa, &items, 50);
            defer selections.deinit(gpa);

            var selected = try zeroOneKnapsackGreedy(gpa, &items, 50);
            defer selected.deinit(gpa);

            try testing.expectEqual(@as(usize, 3), selections.items.len);
            try testing.expectEqualSlices(usize, &.{ 0, 1 }, selected.items);
        }
    };

    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}
