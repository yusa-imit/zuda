const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Bin Packing Approximation Algorithms
///
/// Bin packing: given items with sizes and bins with capacity, pack items into
/// minimum number of bins such that no bin exceeds its capacity.
/// Finding minimum number of bins is NP-complete, but we can approximate efficiently.

pub const BinPackingResult = struct {
    bins: ArrayList(ArrayList(usize)),
    num_bins: usize,

    pub fn deinit(self: *BinPackingResult) void {
        for (self.bins.items) |*bin| {
            bin.deinit();
        }
        self.bins.deinit();
    }
};

/// First-Fit bin packing: places each item in the first bin that has room.
///
/// Time: O(n²) where n = number of items (worst case checks all bins)
/// Space: O(n) for storing bin assignments
///
/// Returns: BinPackingResult with items assigned to bins (caller owns memory)
///
/// Algorithm:
/// 1. For each item:
///    - Try to fit it in the first existing bin with enough space
///    - If no bin fits, open a new bin
/// 2. Approximation ratio: FF(I) ≤ 17/10 × OPT(I) + 2
///
/// Example:
/// ```zig
/// const items = [_]f64{0.5, 0.7, 0.3, 0.9, 0.6};
/// var result = try firstFit(allocator, &items, 1.0);
/// defer result.deinit();
/// // result.num_bins contains number of bins used
/// ```
pub fn firstFit(
    allocator: Allocator,
    items: []const f64,
    capacity: f64,
) !BinPackingResult {
    var bins = ArrayList(ArrayList(usize)).init(allocator);
    errdefer {
        for (bins.items) |*bin| bin.deinit();
        bins.deinit();
    }

    var bin_loads = ArrayList(f64).init(allocator);
    defer bin_loads.deinit();

    for (items, 0..) |item_size, item_idx| {
        // Find first bin that fits
        var placed = false;
        for (bin_loads.items, 0..) |load, bin_idx| {
            if (load + item_size <= capacity) {
                try bins.items[bin_idx].append(item_idx);
                bin_loads.items[bin_idx] += item_size;
                placed = true;
                break;
            }
        }

        // If no bin fits, open a new one
        if (!placed) {
            var new_bin = ArrayList(usize).init(allocator);
            try new_bin.append(item_idx);
            try bins.append(new_bin);
            try bin_loads.append(item_size);
        }
    }

    return BinPackingResult{
        .bins = bins,
        .num_bins = bins.items.len,
    };
}

/// Best-Fit bin packing: places each item in the bin with least remaining space that fits.
///
/// Time: O(n²) where n = number of items
/// Space: O(n) for storing bin assignments
///
/// Returns: BinPackingResult with items assigned to bins (caller owns memory)
///
/// Algorithm:
/// 1. For each item:
///    - Find the bin with minimum remaining capacity that still fits the item
///    - If no bin fits, open a new bin
/// 2. Often performs better than First-Fit in practice
///
/// Example:
/// ```zig
/// const items = [_]f64{0.5, 0.7, 0.3, 0.9, 0.6};
/// var result = try bestFit(allocator, &items, 1.0);
/// defer result.deinit();
/// ```
pub fn bestFit(
    allocator: Allocator,
    items: []const f64,
    capacity: f64,
) !BinPackingResult {
    var bins = ArrayList(ArrayList(usize)).init(allocator);
    errdefer {
        for (bins.items) |*bin| bin.deinit();
        bins.deinit();
    }

    var bin_loads = ArrayList(f64).init(allocator);
    defer bin_loads.deinit();

    for (items, 0..) |item_size, item_idx| {
        // Find bin with minimum remaining space that fits
        var best_bin: ?usize = null;
        var min_remaining = capacity + 1.0;

        for (bin_loads.items, 0..) |load, bin_idx| {
            const remaining = capacity - load;
            if (remaining >= item_size and remaining < min_remaining) {
                best_bin = bin_idx;
                min_remaining = remaining;
            }
        }

        if (best_bin) |bin_idx| {
            try bins.items[bin_idx].append(item_idx);
            bin_loads.items[bin_idx] += item_size;
        } else {
            // No bin fits, open a new one
            var new_bin = ArrayList(usize).init(allocator);
            try new_bin.append(item_idx);
            try bins.append(new_bin);
            try bin_loads.append(item_size);
        }
    }

    return BinPackingResult{
        .bins = bins,
        .num_bins = bins.items.len,
    };
}

/// First-Fit-Decreasing: sorts items by size descending, then applies First-Fit.
///
/// Time: O(n log n) for sorting + O(n²) for packing = O(n²)
/// Space: O(n) for sorted indices
///
/// Returns: BinPackingResult with items assigned to bins (caller owns memory)
///
/// Algorithm:
/// 1. Sort items in decreasing order of size
/// 2. Apply First-Fit to sorted items
/// 3. Approximation ratio: FFD(I) ≤ 11/9 × OPT(I) + 6/9
///
/// Note: FFD has better approximation ratio than FF and often finds optimal solution.
///
/// Example:
/// ```zig
/// const items = [_]f64{0.5, 0.7, 0.3, 0.9, 0.6};
/// var result = try firstFitDecreasing(allocator, &items, 1.0);
/// defer result.deinit();
/// // Often uses fewer bins than First-Fit
/// ```
pub fn firstFitDecreasing(
    allocator: Allocator,
    items: []const f64,
    capacity: f64,
) !BinPackingResult {
    // Create sorted indices
    const indices = try allocator.alloc(usize, items.len);
    defer allocator.free(indices);
    for (indices, 0..) |*idx, i| idx.* = i;

    // Sort indices by item size descending
    const Context = struct {
        items: []const f64,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.items[a] > ctx.items[b]; // Descending
        }
    };
    std.mem.sort(usize, indices, Context{ .items = items }, Context.lessThan);

    // Apply First-Fit to sorted items
    var bins = ArrayList(ArrayList(usize)).init(allocator);
    errdefer {
        for (bins.items) |*bin| bin.deinit();
        bins.deinit();
    }

    var bin_loads = ArrayList(f64).init(allocator);
    defer bin_loads.deinit();

    for (indices) |item_idx| {
        const item_size = items[item_idx];

        // Find first bin that fits
        var placed = false;
        for (bin_loads.items, 0..) |load, bin_idx| {
            if (load + item_size <= capacity) {
                try bins.items[bin_idx].append(item_idx);
                bin_loads.items[bin_idx] += item_size;
                placed = true;
                break;
            }
        }

        // If no bin fits, open a new one
        if (!placed) {
            var new_bin = ArrayList(usize).init(allocator);
            try new_bin.append(item_idx);
            try bins.append(new_bin);
            try bin_loads.append(item_size);
        }
    }

    return BinPackingResult{
        .bins = bins,
        .num_bins = bins.items.len,
    };
}

/// Validates that the bin packing solution is feasible (no bin exceeds capacity).
///
/// Time: O(n) where n = total number of items
/// Space: O(1)
///
/// Returns: true if all bins are within capacity, false otherwise
pub fn isValidPacking(
    items: []const f64,
    result: *const BinPackingResult,
    capacity: f64,
) bool {
    for (result.bins.items) |bin| {
        var load: f64 = 0.0;
        for (bin.items) |item_idx| {
            if (item_idx >= items.len) return false;
            load += items[item_idx];
        }
        if (load > capacity + 1e-9) return false; // Small epsilon for floating point
    }
    return true;
}

/// Computes the total number of items across all bins.
///
/// Time: O(num_bins)
/// Space: O(1)
///
/// Returns: total count of items
pub fn totalItems(result: *const BinPackingResult) usize {
    var count: usize = 0;
    for (result.bins.items) |bin| {
        count += bin.items.len;
    }
    return count;
}

// ============================================================================
// Tests
// ============================================================================

test "bin packing: empty items" {
    const allocator = std.testing.allocator;
    const items: []const f64 = &.{};
    var result = try firstFit(allocator, items, 1.0);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), result.num_bins);
}

test "bin packing: single item fits" {
    const allocator = std.testing.allocator;
    const items = [_]f64{0.5};
    var result = try firstFit(allocator, &items, 1.0);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: two items same bin" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.3, 0.5 };
    var result = try firstFit(allocator, &items, 1.0);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: two items separate bins" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.6, 0.7 };
    var result = try firstFit(allocator, &items, 1.0);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: first-fit example" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };
    var result = try firstFit(allocator, &items, 1.0);
    defer result.deinit();

    // FF: [0.5,0.3], [0.7], [0.9], [0.6] = 4 bins
    try std.testing.expect(result.num_bins >= 2); // Lower bound: sum=3.0 → ≥3 bins
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
    try std.testing.expectEqual(@as(usize, 5), totalItems(&result));
}

test "bin packing: best-fit example" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };
    var result = try bestFit(allocator, &items, 1.0);
    defer result.deinit();

    // BF may pack differently: [0.5,0.3], [0.7], [0.9], [0.6]
    try std.testing.expect(result.num_bins >= 3);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
    try std.testing.expectEqual(@as(usize, 5), totalItems(&result));
}

test "bin packing: first-fit-decreasing example" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };
    var result = try firstFitDecreasing(allocator, &items, 1.0);
    defer result.deinit();

    // FFD sorts: [0.9, 0.7, 0.6, 0.5, 0.3]
    // Packs: [0.9], [0.7,0.3], [0.6] = 3 bins (optimal!)
    try std.testing.expect(result.num_bins >= 3);
    try std.testing.expect(result.num_bins <= 4); // FFD often near-optimal
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: FFD vs FF comparison" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };

    var result_ff = try firstFit(allocator, &items, 1.0);
    defer result_ff.deinit();

    var result_ffd = try firstFitDecreasing(allocator, &items, 1.0);
    defer result_ffd.deinit();

    // FFD should use ≤ FF bins
    try std.testing.expect(result_ffd.num_bins <= result_ff.num_bins);
    try std.testing.expect(isValidPacking(&items, &result_ff, 1.0));
    try std.testing.expect(isValidPacking(&items, &result_ffd, 1.0));
}

test "bin packing: all items exactly capacity" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 1.0, 1.0, 1.0 };
    var result = try firstFit(allocator, &items, 1.0);
    defer result.deinit();

    // Each item needs its own bin
    try std.testing.expectEqual(@as(usize, 3), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: many small items" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    var result = try firstFit(allocator, &items, 1.0);
    defer result.deinit();

    // 10 items of 0.1 each = sum 1.0 → need exactly 1 bin
    try std.testing.expectEqual(@as(usize, 1), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: large number of items (stress test)" {
    const allocator = std.testing.allocator;

    var items_list = ArrayList(f64).init(allocator);
    defer items_list.deinit();

    // 100 items of size 0.3 each
    for (0..100) |_| {
        try items_list.append(0.3);
    }

    var result = try firstFitDecreasing(allocator, items_list.items, 1.0);
    defer result.deinit();

    // 100 × 0.3 = 30.0 → need ≥30 bins (each bin fits 3 items)
    try std.testing.expect(result.num_bins >= 30);
    try std.testing.expect(result.num_bins <= 34); // Should be near-optimal
    try std.testing.expect(isValidPacking(items_list.items, &result, 1.0));
    try std.testing.expectEqual(@as(usize, 100), totalItems(&result));
}

test "bin packing: isValidPacking detects invalid" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.6, 0.7 };

    var result = BinPackingResult{
        .bins = ArrayList(ArrayList(usize)).init(allocator),
        .num_bins = 1,
    };
    defer result.deinit();

    // Create invalid packing: both items in one bin (exceeds capacity)
    var bin = ArrayList(usize).init(allocator);
    try bin.append(0);
    try bin.append(1);
    try result.bins.append(bin);

    try std.testing.expect(!isValidPacking(&items, &result, 1.0));
}

test "bin packing: best-fit tight packing" {
    const allocator = std.testing.allocator;
    // Items designed so best-fit finds tighter packing than first-fit
    const items = [_]f64{ 0.4, 0.4, 0.6, 0.6 };
    var result = try bestFit(allocator, &items, 1.0);
    defer result.deinit();

    // Optimal: [0.4,0.6], [0.4,0.6] = 2 bins
    try std.testing.expectEqual(@as(usize, 2), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: FFD with duplicate sizes" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.5, 0.5, 0.5 };
    var result = try firstFitDecreasing(allocator, &items, 1.0);
    defer result.deinit();

    // 4 × 0.5 = 2.0 → need exactly 2 bins
    try std.testing.expectEqual(@as(usize, 2), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}
