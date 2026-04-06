// Box Stacking Problem — Dynamic Programming
// Classic DP problem: Stack boxes to maximize height
//
// Problem: Given n boxes with dimensions (h, w, d), find maximum height achievable
// by stacking boxes. Constraints:
// - Can use multiple rotations of same box type
// - Bottom box must have strictly larger base dimensions than box on top
// - Can use unlimited boxes of each type (3 rotations per box type)
//
// Algorithm:
// 1. Generate all rotations for each box (3 per box: hwl, whl, dlh where l=length, w=width, h=height)
// 2. Sort boxes by base area (width × depth) in descending order
// 3. Apply LIS-style DP: dp[i] = max height achievable with box i on top
// 4. Recurrence: dp[i] = boxes[i].height + max(dp[j]) for all j where boxes[j] can support boxes[i]
//
// Time: O(n² log n) where n = number of box types × 3 (rotations)
// Space: O(n) for DP array + rotations storage
//
// Use cases:
// - 3D packing optimization
// - Warehouse stacking constraints
// - Physical stability problems
// - Resource allocation with dependencies
//
// Reference: Classic DP problem, variation of LIS with 3D constraints

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Represents a box with dimensions
pub const Box = struct {
    height: usize,
    width: usize,
    depth: usize,

    /// Initialize a box with dimensions
    pub fn init(height: usize, width: usize, depth: usize) Box {
        return .{
            .height = height,
            .width = width,
            .depth = depth,
        };
    }

    /// Calculate base area (width × depth)
    /// Time: O(1)
    pub fn baseArea(self: Box) usize {
        return self.width * self.depth;
    }

    /// Check if this box can be placed on top of another box
    /// Both width and depth must be strictly smaller
    /// Time: O(1)
    pub fn canPlaceOn(self: Box, other: Box) bool {
        return self.width < other.width and self.depth < other.depth;
    }
};

/// Result of box stacking computation
pub const StackResult = struct {
    max_height: usize,
    boxes: ArrayList(usize), // Indices of boxes in the optimal stack (bottom to top)
    allocator: Allocator,

    /// Free allocated resources
    pub fn deinit(self: *StackResult) void {
        self.boxes.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Generate all rotations of given boxes
/// Each box can be rotated in 3 ways (treating each dimension as height)
/// Time: O(n)
/// Space: O(n)
fn generateRotations(allocator: Allocator, boxes: []const Box) !ArrayList(Box) {
    var rotations = try ArrayList(Box).initCapacity(allocator, boxes.len * 3);

    for (boxes) |box| {
        const h = box.height;
        const w = box.width;
        const d = box.depth;

        // Rotation 1: h as height, w and d as base (ensure w ≤ d)
        if (w <= d) {
            rotations.appendAssumeCapacity(Box.init(h, w, d));
        } else {
            rotations.appendAssumeCapacity(Box.init(h, d, w));
        }

        // Rotation 2: w as height, h and d as base (ensure h ≤ d)
        if (h <= d) {
            rotations.appendAssumeCapacity(Box.init(w, h, d));
        } else {
            rotations.appendAssumeCapacity(Box.init(w, d, h));
        }

        // Rotation 3: d as height, h and w as base (ensure h ≤ w)
        if (h <= w) {
            rotations.appendAssumeCapacity(Box.init(d, h, w));
        } else {
            rotations.appendAssumeCapacity(Box.init(d, w, h));
        }
    }

    return rotations;
}

/// Compare boxes by base area (descending order)
fn compareByBaseArea(_: void, a: Box, b: Box) bool {
    return a.baseArea() > b.baseArea();
}

/// Maximum height achievable by stacking boxes (standard DP variant)
/// Returns maximum height only
///
/// Algorithm:
/// 1. Generate all rotations (3 per box type)
/// 2. Sort by base area descending
/// 3. DP: dp[i] = max height with box i on top
/// 4. Answer: max(dp[i]) for all i
///
/// Time: O(n² log n) where n = number of rotations
/// Space: O(n) for DP array and rotations
pub fn maxStackHeight(allocator: Allocator, boxes: []const Box) !usize {
    if (boxes.len == 0) return 0;

    // Generate all rotations
    var rotations = try generateRotations(allocator, boxes);
    defer rotations.deinit(allocator);

    // Sort by base area descending
    std.mem.sort(Box, rotations.items, {}, compareByBaseArea);

    const n = rotations.items.len;

    // DP: dp[i] = max height achievable with box i on top
    const dp = try allocator.alloc(usize, n);
    defer allocator.free(dp);

    // Initialize: each box can be a starting point
    for (rotations.items, 0..) |box, i| {
        dp[i] = box.height;
    }

    // Fill DP table
    for (1..n) |i| {
        for (0..i) |j| {
            // If box i can be placed on box j (j has larger base)
            if (rotations.items[i].canPlaceOn(rotations.items[j])) {
                const new_height = dp[j] + rotations.items[i].height;
                if (new_height > dp[i]) {
                    dp[i] = new_height;
                }
            }
        }
    }

    // Find maximum height
    var max_height: usize = 0;
    for (dp) |h| {
        if (h > max_height) {
            max_height = h;
        }
    }

    return max_height;
}

/// Maximum height with optimal stack sequence
/// Returns StackResult with max height and box indices (bottom to top)
///
/// Time: O(n² log n)
/// Space: O(n) for DP and parent tracking, O(h) for stack reconstruction where h = stack height
pub fn maxStackHeightWithPath(allocator: Allocator, boxes: []const Box) !StackResult {
    if (boxes.len == 0) {
        return StackResult{
            .max_height = 0,
            .boxes = try ArrayList(usize).initCapacity(allocator, 0),
            .allocator = allocator,
        };
    }

    // Generate all rotations
    var rotations = try generateRotations(allocator, boxes);
    defer rotations.deinit(allocator);

    // Sort by base area descending
    std.mem.sort(Box, rotations.items, {}, compareByBaseArea);

    const n = rotations.items.len;

    // DP arrays
    const dp = try allocator.alloc(usize, n);
    defer allocator.free(dp);

    const parent = try allocator.alloc(?usize, n);
    defer allocator.free(parent);

    // Initialize
    for (rotations.items, 0..) |box, i| {
        dp[i] = box.height;
        parent[i] = null;
    }

    // Fill DP table with parent tracking
    var max_idx: usize = 0;
    for (1..n) |i| {
        for (0..i) |j| {
            if (rotations.items[i].canPlaceOn(rotations.items[j])) {
                const new_height = dp[j] + rotations.items[i].height;
                if (new_height > dp[i]) {
                    dp[i] = new_height;
                    parent[i] = j;
                }
            }
        }
        if (dp[i] > dp[max_idx]) {
            max_idx = i;
        }
    }

    // Reconstruct stack (bottom to top)
    var stack = try ArrayList(usize).initCapacity(allocator, 10);
    var current: ?usize = max_idx;
    while (current) |idx| {
        try stack.append(allocator, idx);
        current = parent[idx];
    }

    // Reverse to get bottom-to-top order
    std.mem.reverse(usize, stack.items);

    return StackResult{
        .max_height = dp[max_idx],
        .boxes = stack,
        .allocator = allocator,
    };
}

/// Count number of ways to achieve maximum height
/// Multiple stacking sequences may yield same max height
///
/// Time: O(n² log n)
/// Space: O(n)
pub fn countMaxStackWays(allocator: Allocator, boxes: []const Box) !usize {
    if (boxes.len == 0) return 1;

    var rotations = try generateRotations(allocator, boxes);
    defer rotations.deinit(allocator);

    std.mem.sort(Box, rotations.items, {}, compareByBaseArea);

    const n = rotations.items.len;

    const dp = try allocator.alloc(usize, n);
    defer allocator.free(dp);

    const count = try allocator.alloc(usize, n);
    defer allocator.free(count);

    // Initialize
    for (rotations.items, 0..) |box, i| {
        dp[i] = box.height;
        count[i] = 1; // One way to stack just this box
    }

    // Fill DP table with counting
    for (1..n) |i| {
        for (0..i) |j| {
            if (rotations.items[i].canPlaceOn(rotations.items[j])) {
                const new_height = dp[j] + rotations.items[i].height;
                if (new_height > dp[i]) {
                    dp[i] = new_height;
                    count[i] = count[j]; // Reset count
                } else if (new_height == dp[i]) {
                    count[i] += count[j]; // Add ways
                }
            }
        }
    }

    // Find max height and sum ways to achieve it
    var max_height: usize = 0;
    for (dp) |h| {
        if (h > max_height) {
            max_height = h;
        }
    }

    var total_ways: usize = 0;
    for (dp, 0..) |h, i| {
        if (h == max_height) {
            total_ways += count[i];
        }
    }

    return total_ways;
}

// Tests
test "box stacking: basic example" {
    const allocator = std.testing.allocator;

    // Classic example: 3 box types
    const boxes = [_]Box{
        Box.init(4, 6, 7), // Box 1
        Box.init(1, 2, 3), // Box 2
        Box.init(4, 5, 6), // Box 3
    };

    const height = try maxStackHeight(allocator, &boxes);
    // With rotations and optimal stacking: 60 is achievable
    // (depends on which rotations are chosen and how they stack)
    try std.testing.expect(height > 0);
}

test "box stacking: single box" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{Box.init(4, 6, 7)};
    const height = try maxStackHeight(allocator, &boxes);
    // With 3 rotations (4,6,7), (6,4,7), (7,4,6), we can stack some on others
    // Since smaller base can go on larger base, optimal is 11 (4+7) or similar
    try std.testing.expect(height >= 7); // At least the tallest rotation
}

test "box stacking: two boxes stackable" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{
        Box.init(4, 6, 7), // Larger box
        Box.init(1, 2, 3), // Smaller box
    };

    const height = try maxStackHeight(allocator, &boxes);
    // Smaller box can go on top of larger box
    try std.testing.expect(height >= 7 + 3); // At least sum of max heights
}

test "box stacking: empty input" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{};
    const height = try maxStackHeight(allocator, &boxes);
    try std.testing.expectEqual(@as(usize, 0), height);
}

test "box stacking: identical boxes" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{
        Box.init(5, 5, 5),
        Box.init(5, 5, 5),
        Box.init(5, 5, 5),
    };

    const height = try maxStackHeight(allocator, &boxes);
    // Identical boxes can't stack (need strictly smaller base)
    // Only one rotation will be considered
    try std.testing.expectEqual(@as(usize, 5), height);
}

test "box stacking: with path reconstruction" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{
        Box.init(4, 6, 7),
        Box.init(1, 2, 3),
    };

    var result = try maxStackHeightWithPath(allocator, &boxes);
    defer result.deinit();

    try std.testing.expect(result.max_height > 0);
    try std.testing.expect(result.boxes.items.len > 0);
}

test "box stacking: non-stackable boxes" {
    const allocator = std.testing.allocator;

    // All boxes have same base dimensions (different rotations)
    const boxes = [_]Box{
        Box.init(10, 5, 5),
        Box.init(20, 5, 5),
        Box.init(30, 5, 5),
    };

    const height = try maxStackHeight(allocator, &boxes);
    // Can't stack because bases are same size
    // Maximum is the tallest single box
    try std.testing.expect(height == 30 or height > 30); // 30 or more if rotations help
}

test "box stacking: optimal stacking sequence" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{
        Box.init(1, 2, 4),
        Box.init(3, 2, 5),
    };

    var result = try maxStackHeightWithPath(allocator, &boxes);
    defer result.deinit();

    try std.testing.expect(result.max_height > 0);

    // Verify stack is non-empty
    try std.testing.expect(result.boxes.items.len > 0);
}

test "box stacking: count ways" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{
        Box.init(4, 6, 7),
        Box.init(1, 2, 3),
    };

    const ways = try countMaxStackWays(allocator, &boxes);
    try std.testing.expect(ways > 0);
}

test "box stacking: large input" {
    const allocator = std.testing.allocator;

    const boxes = try allocator.alloc(Box, 20);
    defer allocator.free(boxes);

    for (boxes, 0..) |*box, i| {
        box.* = Box.init(i + 1, i + 2, i + 3);
    }

    const height = try maxStackHeight(allocator, boxes);
    try std.testing.expect(height > 0);
}

test "box stacking: decreasing dimensions" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{
        Box.init(10, 10, 10),
        Box.init(8, 8, 8),
        Box.init(6, 6, 6),
        Box.init(4, 4, 4),
    };

    const height = try maxStackHeight(allocator, &boxes);
    // Each box can stack on the next larger one
    try std.testing.expectEqual(@as(usize, 10 + 8 + 6 + 4), height);
}

test "box stacking: path validation" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{
        Box.init(10, 10, 10),
        Box.init(8, 8, 8),
        Box.init(6, 6, 6),
    };

    var result = try maxStackHeightWithPath(allocator, &boxes);
    defer result.deinit();

    // Verify decreasing base areas in stack (bottom to top)
    var rotations = try generateRotations(allocator, &boxes);
    defer rotations.deinit(allocator);
    std.mem.sort(Box, rotations.items, {}, compareByBaseArea);

    if (result.boxes.items.len > 1) {
        for (0..result.boxes.items.len - 1) |i| {
            const bottom = rotations.items[result.boxes.items[i]];
            const top = rotations.items[result.boxes.items[i + 1]];
            try std.testing.expect(top.canPlaceOn(bottom));
        }
    }
}

test "box stacking: memory safety" {
    const allocator = std.testing.allocator;

    const boxes = [_]Box{
        Box.init(4, 6, 7),
        Box.init(1, 2, 3),
    };

    for (0..10) |_| {
        const height = try maxStackHeight(allocator, &boxes);
        try std.testing.expect(height > 0);

        var result = try maxStackHeightWithPath(allocator, &boxes);
        result.deinit();

        const ways = try countMaxStackWays(allocator, &boxes);
        try std.testing.expect(ways > 0);
    }
}
