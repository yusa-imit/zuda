//! Bin packing approximation algorithms.
//!
//! Bin packing: given items with sizes and bins with a capacity, pack the items into as few
//! bins as possible so that no bin exceeds its capacity. The optimum is NP-complete; First-Fit,
//! Best-Fit and First-Fit-Decreasing are fast approximations.
//!
//! An item larger than `capacity` cannot be packed feasibly; it is placed alone in its own bin
//! and `isValidPacking` reports the result as infeasible.
//!
//! A negative or NaN capacity and a negative or NaN item size are user data and return
//! `error.InvalidInput` before anything is allocated.
//!
//! Ownership: a `BinPackingResult` owns its bins; release it with `deinit(gpa)` using the
//! allocator that produced it. On error nothing is leaked.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

pub const Error = error{InvalidInput} || Allocator.Error;

pub const BinPackingResult = struct {
    bins: ArrayList(ArrayList(usize)),
    num_bins: usize,

    /// Releases every bin. `gpa` must be the allocator that produced the result.
    pub fn deinit(self: *BinPackingResult, gpa: Allocator) void {
        for (self.bins.items) |*bin| {
            bin.deinit(gpa);
        }
        self.bins.deinit(gpa);
        self.* = undefined;
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
/// var result = try firstFit(gpa, &items, 1.0);
/// defer result.deinit(gpa);
/// // result.num_bins contains number of bins used
/// ```
pub fn firstFit(
    gpa: Allocator,
    items: []const f64,
    capacity: f64,
) Error!BinPackingResult {
    try input_validate(items, capacity);
    assert(capacity >= 0.0);

    var packing = try Packing.init(gpa, items.len);
    errdefer packing.deinit(gpa);

    for (items, 0..) |item_size, item_idx| {
        try packing.placeFirstFit(gpa, item_idx, item_size, capacity);
    }

    return packing.finish(gpa);
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
/// var result = try bestFit(gpa, &items, 1.0);
/// defer result.deinit(gpa);
/// ```
pub fn bestFit(
    gpa: Allocator,
    items: []const f64,
    capacity: f64,
) Error!BinPackingResult {
    try input_validate(items, capacity);
    assert(capacity >= 0.0);

    var packing = try Packing.init(gpa, items.len);
    errdefer packing.deinit(gpa);

    for (items, 0..) |item_size, item_idx| {
        // Find bin with minimum remaining space that fits
        var best_bin: ?usize = null;
        var min_remaining = std.math.inf(f64);

        for (packing.loads.items, 0..) |load, bin_idx| {
            const remaining = capacity - load;
            if (remaining >= item_size and remaining < min_remaining) {
                best_bin = bin_idx;
                min_remaining = remaining;
            }
        }

        if (best_bin) |bin_idx| {
            try packing.addToBin(gpa, bin_idx, item_idx, item_size);
        } else {
            try packing.openBin(gpa, item_idx, item_size);
        }
    }

    return packing.finish(gpa);
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
/// var result = try firstFitDecreasing(gpa, &items, 1.0);
/// defer result.deinit(gpa);
/// // Often uses fewer bins than First-Fit
/// ```
pub fn firstFitDecreasing(
    gpa: Allocator,
    items: []const f64,
    capacity: f64,
) Error!BinPackingResult {
    try input_validate(items, capacity);
    assert(capacity >= 0.0);

    // Create sorted indices
    const indices = try gpa.alloc(usize, items.len);
    defer gpa.free(indices);

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
    var packing = try Packing.init(gpa, items.len);
    errdefer packing.deinit(gpa);

    for (indices) |item_idx| {
        try packing.placeFirstFit(gpa, item_idx, items[item_idx], capacity);
    }

    return packing.finish(gpa);
}

/// Returns `error.InvalidInput` for a negative or NaN capacity or item size.
fn input_validate(items: []const f64, capacity: f64) error{InvalidInput}!void {
    // Written as `!(x >= 0)` so that NaN is rejected too.
    if (!(capacity >= 0.0)) return error.InvalidInput;
    for (items) |item_size| {
        if (!(item_size >= 0.0)) return error.InvalidInput;
    }
}

/// Bins under construction plus their running loads; shared by all three algorithms.
const Packing = struct {
    bins: ArrayList(ArrayList(usize)),
    loads: ArrayList(f64),

    /// Reserves room for `bins_max` bins up front (never more bins than items).
    fn init(gpa: Allocator, bins_max: usize) !Packing {
        var packing = Packing{ .bins = .empty, .loads = .empty };
        errdefer packing.deinit(gpa);

        try packing.bins.ensureTotalCapacityPrecise(gpa, bins_max);
        try packing.loads.ensureTotalCapacityPrecise(gpa, bins_max);
        return packing;
    }

    fn deinit(packing: *Packing, gpa: Allocator) void {
        assert(packing.bins.items.len == packing.loads.items.len);
        for (packing.bins.items) |*bin| bin.deinit(gpa);
        packing.bins.deinit(gpa);
        packing.loads.deinit(gpa);
    }

    /// Hands the bins to the caller; the load scratch is released.
    fn finish(packing: *Packing, gpa: Allocator) BinPackingResult {
        assert(packing.bins.items.len == packing.loads.items.len);
        const num_bins = packing.bins.items.len;
        packing.loads.deinit(gpa);
        return BinPackingResult{ .bins = packing.bins, .num_bins = num_bins };
    }

    fn placeFirstFit(
        packing: *Packing,
        gpa: Allocator,
        item_idx: usize,
        item_size: f64,
        capacity: f64,
    ) !void {
        for (packing.loads.items, 0..) |load, bin_idx| {
            if (load + item_size <= capacity) {
                return packing.addToBin(gpa, bin_idx, item_idx, item_size);
            }
        }
        return packing.openBin(gpa, item_idx, item_size);
    }

    fn addToBin(
        packing: *Packing,
        gpa: Allocator,
        bin_idx: usize,
        item_idx: usize,
        item_size: f64,
    ) !void {
        assert(bin_idx < packing.bins.items.len);
        try packing.bins.items[bin_idx].append(gpa, item_idx);
        packing.loads.items[bin_idx] += item_size;
    }

    /// Opens a new bin holding one item. Capacity was reserved in `init`, so only the bin's
    /// own item list allocates, and a failure there leaves no half-built bin behind.
    fn openBin(packing: *Packing, gpa: Allocator, item_idx: usize, item_size: f64) !void {
        assert(packing.bins.items.len < packing.bins.capacity);
        assert(packing.loads.items.len < packing.loads.capacity);

        var new_bin: ArrayList(usize) = .empty;
        try new_bin.append(gpa, item_idx);
        packing.bins.appendAssumeCapacity(new_bin);
        packing.loads.appendAssumeCapacity(item_size);
    }
};

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
    assert(result.num_bins == result.bins.items.len);
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
    assert(result.num_bins == result.bins.items.len);
    var count: usize = 0;
    for (result.bins.items) |bin| {
        count += bin.items.len;
    }
    assert(count == 0 or result.bins.items.len > 0);
    return count;
}

// ============================================================================
// Tests
// ============================================================================

test "bin packing: empty items" {
    const gpa = std.testing.allocator;
    const items: []const f64 = &.{};
    var result = try firstFit(gpa, items, 1.0);
    defer result.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), result.num_bins);
}

test "bin packing: single item fits" {
    const gpa = std.testing.allocator;
    const items = [_]f64{0.5};
    var result = try firstFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: two items same bin" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.3, 0.5 };
    var result = try firstFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: two items separate bins" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.6, 0.7 };
    var result = try firstFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 2), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: first-fit example" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };
    var result = try firstFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    // FF: [0.5,0.3], [0.7], [0.9], [0.6] = 4 bins
    try std.testing.expectEqual(@as(usize, 4), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
    try std.testing.expectEqual(@as(usize, 5), totalItems(&result));
}

test "bin packing: best-fit example" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };
    var result = try bestFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    // BF puts 0.3 into the fuller bin of 0.7: [0.5], [0.7,0.3], [0.9], [0.6] = 4 bins
    try std.testing.expectEqual(@as(usize, 4), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
    try std.testing.expectEqual(@as(usize, 5), totalItems(&result));
}

test "bin packing: first-fit-decreasing example" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };
    var result = try firstFitDecreasing(gpa, &items, 1.0);
    defer result.deinit(gpa);

    // FFD sorts: [0.9, 0.7, 0.6, 0.5, 0.3]
    // Packs: [0.9], [0.7,0.3], [0.6], [0.5] = 4 bins. Every bin of 3 total size would have to
    // be exactly full, which 0.9 forbids, so 4 is optimal.
    try std.testing.expectEqual(@as(usize, 4), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: FFD vs FF comparison" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };

    var result_ff = try firstFit(gpa, &items, 1.0);
    defer result_ff.deinit(gpa);

    var result_ffd = try firstFitDecreasing(gpa, &items, 1.0);
    defer result_ffd.deinit(gpa);

    // FFD should use ≤ FF bins
    try std.testing.expect(result_ffd.num_bins <= result_ff.num_bins);
    try std.testing.expect(isValidPacking(&items, &result_ff, 1.0));
    try std.testing.expect(isValidPacking(&items, &result_ffd, 1.0));
}

test "bin packing: all items exactly capacity" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 1.0, 1.0, 1.0 };
    var result = try firstFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    // Each item needs its own bin
    try std.testing.expectEqual(@as(usize, 3), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: many small items" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    var result = try firstFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    // 10 items of 0.1 each = sum 1.0 → need exactly 1 bin
    try std.testing.expectEqual(@as(usize, 1), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: large number of items (stress test)" {
    const gpa = std.testing.allocator;

    var items_list: ArrayList(f64) = .empty;
    defer items_list.deinit(gpa);

    // 100 items of size 0.3 each
    for (0..100) |_| {
        try items_list.append(gpa, 0.3);
    }

    var result = try firstFitDecreasing(gpa, items_list.items, 1.0);
    defer result.deinit(gpa);

    // Each bin fits 3 items (4 × 0.3 > 1.0): 33 full bins plus one holding the last item.
    try std.testing.expectEqual(@as(usize, 34), result.num_bins);
    try std.testing.expect(isValidPacking(items_list.items, &result, 1.0));
    try std.testing.expectEqual(@as(usize, 100), totalItems(&result));
}

test "bin packing: isValidPacking detects invalid" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.6, 0.7 };

    var result = BinPackingResult{
        .bins = .empty,
        .num_bins = 1,
    };
    defer result.deinit(gpa);

    // Create invalid packing: both items in one bin (exceeds capacity)
    var bin: ArrayList(usize) = .empty;
    try bin.append(gpa, 0);
    try bin.append(gpa, 1);
    try result.bins.append(gpa, bin);

    try std.testing.expect(!isValidPacking(&items, &result, 1.0));
}

test "bin packing: best-fit is online and can miss the optimum" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.4, 0.4, 0.6, 0.6 };
    var result = try bestFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    // Online order: [0.4,0.4], [0.6], [0.6] = 3 bins, although [0.4,0.6] twice fits in 2.
    try std.testing.expectEqual(@as(usize, 3), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: best-fit prefers the fuller bin" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.7, 0.3 };
    var best = try bestFit(gpa, &items, 1.0);
    defer best.deinit(gpa);

    var first = try firstFit(gpa, &items, 1.0);
    defer first.deinit(gpa);

    // 0.3 fits both open bins; Best-Fit takes the fuller one (0.7), First-Fit the first (0.5).
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, best.bins.items[1].items);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, first.bins.items[0].items);
}

test "bin packing: FFD with duplicate sizes" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.5, 0.5, 0.5, 0.5 };
    var result = try firstFitDecreasing(gpa, &items, 1.0);
    defer result.deinit(gpa);

    // 4 × 0.5 = 2.0 → need exactly 2 bins
    try std.testing.expectEqual(@as(usize, 2), result.num_bins);
    try std.testing.expect(isValidPacking(&items, &result, 1.0));
}

test "bin packing: best-fit with a capacity far above the item sizes" {
    const gpa = std.testing.allocator;
    // `capacity + 1.0` rounds back to `capacity` at 1e20, which used to reject every bin.
    const items = [_]f64{ 1.0, 1.0 };
    var result = try bestFit(gpa, &items, 1.0e20);
    defer result.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), result.num_bins);
    try std.testing.expectEqual(@as(usize, 2), totalItems(&result));
}

test "bin packing: oversized item gets its own bin and fails validation" {
    const gpa = std.testing.allocator;
    const items = [_]f64{ 0.5, 2.0 };
    var result = try firstFit(gpa, &items, 1.0);
    defer result.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 2), result.num_bins);
    try std.testing.expect(!isValidPacking(&items, &result, 1.0));
}

test "bin packing: no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const items = [_]f64{ 0.5, 0.7, 0.3, 0.9, 0.6 };
            var first = try firstFit(gpa, &items, 1.0);
            defer first.deinit(gpa);

            var best = try bestFit(gpa, &items, 1.0);
            defer best.deinit(gpa);

            var decreasing = try firstFitDecreasing(gpa, &items, 1.0);
            defer decreasing.deinit(gpa);

            try std.testing.expect(isValidPacking(&items, &first, 1.0));
            try std.testing.expect(isValidPacking(&items, &best, 1.0));
            try std.testing.expect(isValidPacking(&items, &decreasing, 1.0));
        }
    };

    try std.testing.checkAllAllocationFailures(std.testing.allocator, Runner.run, .{});
}

test "bin packing: invalid capacity or item size is InvalidInput in all algorithms" {
    const gpa = std.testing.allocator;
    const nan = std.math.nan(f64);
    const good = [_]f64{ 0.5, 0.4 };
    const negative_item = [_]f64{ 0.5, -0.1 };
    const nan_item = [_]f64{ 0.5, nan };

    try std.testing.expectError(error.InvalidInput, firstFit(gpa, &good, -1.0));
    try std.testing.expectError(error.InvalidInput, bestFit(gpa, &good, -1.0));
    try std.testing.expectError(error.InvalidInput, firstFitDecreasing(gpa, &good, -1.0));
    try std.testing.expectError(error.InvalidInput, firstFit(gpa, &good, nan));
    try std.testing.expectError(error.InvalidInput, bestFit(gpa, &good, nan));
    try std.testing.expectError(error.InvalidInput, firstFitDecreasing(gpa, &good, nan));
    try std.testing.expectError(error.InvalidInput, firstFit(gpa, &negative_item, 1.0));
    try std.testing.expectError(error.InvalidInput, bestFit(gpa, &negative_item, 1.0));
    try std.testing.expectError(error.InvalidInput, firstFitDecreasing(gpa, &nan_item, 1.0));
    try std.testing.expectError(error.InvalidInput, firstFit(gpa, &nan_item, 1.0));

    // Zero capacity and zero-size items are valid boundaries.
    const zero_items = [_]f64{0.0};
    var result = try firstFit(gpa, &zero_items, 0.0);
    defer result.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), result.num_bins);
}
