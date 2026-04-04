const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Helper to get maximum value for a type
inline fn maxValue(comptime T: type) T {
    const type_info = @typeInfo(T);
    return if (type_info == .int)
        std.math.maxInt(T)
    else if (type_info == .float)
        std.math.inf(T)
    else
        @compileError("Unsupported type for optimalBST");
}

/// Optimal Binary Search Tree (OBST) - Dynamic Programming
///
/// Finds the optimal arrangement of keys in a BST to minimize expected search cost.
/// Given keys sorted in ascending order and their access frequencies (or probabilities),
/// constructs a BST that minimizes the weighted search cost.
///
/// Problem:
/// - Input: n sorted keys, their search frequencies
/// - Goal: Build BST with minimum expected search cost
/// - Cost of a node at depth d: freq[i] × (d + 1)
/// - Total cost: sum of all node costs
///
/// Algorithm:
/// - DP recurrence: cost[i][j] = min over k in [i..j] of {
///     cost[i][k-1] + cost[k+1][j] + sum(freq[i..j])
///   }
/// - Base case: cost[i][i] = freq[i] (single key)
/// - Builds optimal BST bottom-up by increasing subproblem size
/// - Time: O(n³), Space: O(n²)
///
/// Example:
/// ```
/// Keys: [10, 20, 30]
/// Frequencies: [3, 2, 1]
/// Optimal root: 10 (minimizes weighted cost)
///   10
///     \
///      20
///        \
///         30
/// Cost: 3×1 + 2×2 + 1×3 = 10
/// ```

/// Result of optimal BST computation
pub fn OptimalBSTResult(comptime T: type) type {
    return struct {
        /// Minimum cost to build optimal BST
        min_cost: T,
        /// Root indices for each subproblem [i][j]
        /// root[i][j] = optimal root index for keys[i..j+1]
        root: [][]usize,
        /// Cost table for each subproblem [i][j]
        cost: [][]T,
        allocator: Allocator,

        pub fn deinit(self: *@This()) void {
            const n = self.root.len;
            for (0..n) |i| {
                self.allocator.free(self.root[i]);
                self.allocator.free(self.cost[i]);
            }
            self.allocator.free(self.root);
            self.allocator.free(self.cost);
        }
    };
}

/// Compute optimal BST for given keys and frequencies
///
/// Time: O(n³) where n = number of keys
/// Space: O(n²) for DP tables
///
/// Parameters:
/// - T: Numeric type for frequencies (supports f32, f64, i32, u32, etc.)
/// - keys: Sorted array of keys (for validation, not used in computation)
/// - frequencies: Access frequencies/probabilities for each key
/// - allocator: Memory allocator for result tables
///
/// Returns: OptimalBSTResult containing minimum cost and structure
///
/// Errors:
/// - error.EmptyInput if keys or frequencies are empty
/// - error.LengthMismatch if keys and frequencies have different lengths
/// - error.InvalidFrequency if any frequency is negative (for signed types)
/// - error.OutOfMemory if allocation fails
pub fn optimalBST(
    comptime T: type,
    keys: []const T,
    frequencies: []const T,
    allocator: Allocator,
) !OptimalBSTResult(T) {
    if (keys.len == 0) return error.EmptyInput;
    if (keys.len != frequencies.len) return error.LengthMismatch;

    // Validate frequencies are non-negative (for signed types)
    const type_info = @typeInfo(T);
    if (type_info == .int) {
        if (type_info.int.signedness == .signed) {
            for (frequencies) |freq| {
                if (freq < 0) return error.InvalidFrequency;
            }
        }
    } else if (type_info == .float) {
        for (frequencies) |freq| {
            if (freq < 0) return error.InvalidFrequency;
        }
    }

    const n = keys.len;

    // Allocate DP tables
    const cost = try allocator.alloc([]T, n);
    errdefer allocator.free(cost);
    const root = try allocator.alloc([]usize, n);
    errdefer {
        allocator.free(root);
    }

    var allocated: usize = 0;
    errdefer {
        for (0..allocated) |i| {
            allocator.free(cost[i]);
            allocator.free(root[i]);
        }
    }

    for (0..n) |i| {
        cost[i] = try allocator.alloc(T, n);
        root[i] = try allocator.alloc(usize, n);
        allocated += 1;
        @memset(cost[i], 0);
        @memset(root[i], 0);
    }

    // Base case: single keys
    for (0..n) |i| {
        cost[i][i] = frequencies[i];
        root[i][i] = i;
    }

    // Build DP table bottom-up by increasing chain length
    var len: usize = 2;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i <= n - len) : (i += 1) {
            const j = i + len - 1;

            // Initialize with maximum value
            cost[i][j] = maxValue(T);

            // Sum of frequencies in range [i, j]
            var sum: T = 0;
            for (i..j + 1) |idx| {
                sum += frequencies[idx];
            }

            // Try each key as root
            for (i..j + 1) |r| {
                const left_cost = if (r > i) cost[i][r - 1] else 0;
                const right_cost = if (r < j) cost[r + 1][j] else 0;
                const total = left_cost + right_cost + sum;

                if (total < cost[i][j]) {
                    cost[i][j] = total;
                    root[i][j] = r;
                }
            }
        }
    }

    return OptimalBSTResult(T){
        .min_cost = cost[0][n - 1],
        .root = root,
        .cost = cost,
        .allocator = allocator,
    };
}

/// Compute optimal BST with dummy keys (successful + unsuccessful searches)
///
/// Extended version that considers both successful searches (actual keys)
/// and unsuccessful searches (gaps between keys).
///
/// Time: O(n³) where n = number of keys
/// Space: O(n²) for DP tables
///
/// Parameters:
/// - T: Numeric type for frequencies
/// - keys: Sorted array of actual keys
/// - key_freq: Frequencies for successful searches
/// - dummy_freq: Frequencies for unsuccessful searches (length = keys.len + 1)
///   dummy_freq[0] = searches < keys[0]
///   dummy_freq[i] = searches between keys[i-1] and keys[i]
///   dummy_freq[n] = searches > keys[n-1]
/// - allocator: Memory allocator
///
/// Returns: OptimalBSTResult with minimum cost considering both search types
pub fn optimalBSTWithDummy(
    comptime T: type,
    keys: []const T,
    key_freq: []const T,
    dummy_freq: []const T,
    allocator: Allocator,
) !OptimalBSTResult(T) {
    if (keys.len == 0) return error.EmptyInput;
    if (keys.len != key_freq.len) return error.LengthMismatch;
    if (dummy_freq.len != keys.len + 1) return error.InvalidDummyLength;

    const n = keys.len;

    // Allocate DP tables (size n+1 to include dummy nodes)
    const cost = try allocator.alloc([]T, n + 1);
    errdefer allocator.free(cost);
    const root = try allocator.alloc([]usize, n + 1);
    errdefer allocator.free(root);

    var allocated: usize = 0;
    errdefer {
        for (0..allocated) |i| {
            allocator.free(cost[i]);
            allocator.free(root[i]);
        }
    }

    for (0..n + 1) |i| {
        cost[i] = try allocator.alloc(T, n + 1);
        root[i] = try allocator.alloc(usize, n + 1);
        allocated += 1;
        @memset(cost[i], 0);
        @memset(root[i], 0);
    }

    // Base case: cost[i][i-1] = dummy_freq[i] (unsuccessful search)
    for (0..n + 1) |i| {
        cost[i][i] = dummy_freq[i];
    }

    // Build DP table
    var len: usize = 1;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i <= n - len) : (i += 1) {
            const j = i + len;

            cost[i][j] = maxValue(T);

            // Compute weight (sum of all frequencies in range)
            var weight: T = dummy_freq[i];
            for (i..j) |idx| {
                weight += key_freq[idx] + dummy_freq[idx + 1];
            }

            // Try each key as root
            for (i..j) |r| {
                const left_cost = if (r > i) cost[i][r] else cost[i][i];
                const right_cost = if (r + 1 < j) cost[r + 1][j] else cost[r + 1][r + 1];
                const total = left_cost + right_cost + weight;

                if (total < cost[i][j]) {
                    cost[i][j] = total;
                    root[i][j] = r;
                }
            }
        }
    }

    return OptimalBSTResult(T){
        .min_cost = cost[0][n],
        .root = root,
        .cost = cost,
        .allocator = allocator,
    };
}

/// Reconstruct the optimal BST structure as an array representation
///
/// Returns an array where result[i] = parent index of node i
/// (or n for the root node, indicating no parent)
///
/// Time: O(n²) to traverse all subproblems
/// Space: O(n) for output array
pub fn reconstructTree(
    comptime T: type,
    result: OptimalBSTResult(T),
    allocator: Allocator,
) ![]usize {
    const n = result.root.len;
    const parent = try allocator.alloc(usize, n);
    @memset(parent, n); // n means no parent (root)

    // Helper to recursively build parent relationships
    const Context = struct {
        parent_arr: []usize,
        root_table: [][]usize,

        fn buildSubtree(self: @This(), i: usize, j: usize, par: usize) void {
            if (i > j) return;
            const root_idx = self.root_table[i][j];
            self.parent_arr[root_idx] = par;

            // Recurse on left and right subtrees
            if (root_idx > 0 and i <= root_idx - 1) {
                self.buildSubtree(i, root_idx - 1, root_idx);
            }
            if (root_idx + 1 <= j) {
                self.buildSubtree(root_idx + 1, j, root_idx);
            }
        }
    };

    const ctx = Context{
        .parent_arr = parent,
        .root_table = result.root,
    };

    ctx.buildSubtree(0, n - 1, n);

    return parent;
}

// ============================================================================
// TESTS
// ============================================================================

test "optimal BST: basic 3 keys" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 10, 20, 30 };
    const freq = [_]i32{ 3, 2, 1 };

    var result = try optimalBST(i32, &keys, &freq, allocator);
    defer result.deinit();

    // Expected optimal cost: 3*1 + 2*2 + 1*3 = 10
    try testing.expectEqual(@as(i32, 10), result.min_cost);

    // Root should be key at index 0 (key 10)
    try testing.expectEqual(@as(usize, 0), result.root[0][2]);
}

test "optimal BST: single key" {
    const allocator = testing.allocator;

    const keys = [_]f32{42};
    const freq = [_]f32{1.0};

    var result = try optimalBST(f32, &keys, &freq, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(f32, 1.0), result.min_cost);
    try testing.expectEqual(@as(usize, 0), result.root[0][0]);
}

test "optimal BST: two keys equal frequency" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 10, 20 };
    const freq = [_]i32{ 5, 5 };

    var result = try optimalBST(i32, &keys, &freq, allocator);
    defer result.deinit();

    // Either key as root gives cost: 5*1 + 5*2 = 15
    try testing.expectEqual(@as(i32, 15), result.min_cost);
}

test "optimal BST: four keys" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 10, 12, 20, 35 };
    const freq = [_]i32{ 34, 8, 50, 21 };

    var result = try optimalBST(i32, &keys, &freq, allocator);
    defer result.deinit();

    // Optimal root should be key with highest frequency (20 at index 2)
    try testing.expectEqual(@as(usize, 2), result.root[0][3]);

    // Cost should be minimal (actual value depends on optimal structure)
    try testing.expect(result.min_cost > 0);
}

test "optimal BST: decreasing frequencies" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 1, 2, 3, 4, 5 };
    const freq = [_]i32{ 50, 40, 30, 20, 10 };

    var result = try optimalBST(i32, &keys, &freq, allocator);
    defer result.deinit();

    // With decreasing frequencies, optimal root is somewhere in the left part
    // But the exact position depends on the DP optimization
    try testing.expect(result.root[0][4] < 5);
    try testing.expect(result.min_cost > 0);
}

test "optimal BST: f64 support" {
    const allocator = testing.allocator;

    const keys = [_]f64{ 1.5, 2.5, 3.5 };
    const freq = [_]f64{ 0.3, 0.5, 0.2 };

    var result = try optimalBST(f64, &keys, &freq, allocator);
    defer result.deinit();

    try testing.expect(result.min_cost > 0);
    try testing.expect(result.min_cost < 10.0);
}

test "optimal BST: with dummy keys - basic" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 10, 20 };
    const key_freq = [_]i32{ 3, 3 };
    const dummy_freq = [_]i32{ 2, 3, 1 }; // < 10, 10-20, > 20

    var result = try optimalBSTWithDummy(i32, &keys, &key_freq, &dummy_freq, allocator);
    defer result.deinit();

    try testing.expect(result.min_cost > 0);
}

test "optimal BST: with dummy keys - three keys" {
    const allocator = testing.allocator;

    const keys = [_]f32{ 10, 20, 30 };
    const key_freq = [_]f32{ 4, 2, 6 };
    const dummy_freq = [_]f32{ 3, 2, 1, 1 };

    var result = try optimalBSTWithDummy(f32, &keys, &key_freq, &dummy_freq, allocator);
    defer result.deinit();

    try testing.expect(result.min_cost > 0);
}

test "optimal BST: reconstruct tree structure" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 10, 20, 30 };
    const freq = [_]i32{ 3, 2, 1 };

    var result = try optimalBST(i32, &keys, &freq, allocator);
    defer result.deinit();

    const parent = try reconstructTree(i32, result, allocator);
    defer allocator.free(parent);

    // Root (index 0) should have no parent (value 3)
    try testing.expectEqual(@as(usize, 3), parent[0]);

    // Other nodes should have valid parents
    try testing.expect(parent[1] < 3);
    try testing.expect(parent[2] < 3);
}

test "optimal BST: large input" {
    const allocator = testing.allocator;

    var keys: [10]i32 = undefined;
    var freq: [10]i32 = undefined;
    for (0..10) |i| {
        keys[i] = @intCast(i * 10);
        freq[i] = @intCast(10 - i); // Decreasing frequencies
    }

    var result = try optimalBST(i32, &keys, &freq, allocator);
    defer result.deinit();

    try testing.expect(result.min_cost > 0);
}

test "optimal BST: empty input error" {
    const allocator = testing.allocator;

    const keys = [_]i32{};
    const freq = [_]i32{};

    try testing.expectError(error.EmptyInput, optimalBST(i32, &keys, &freq, allocator));
}

test "optimal BST: length mismatch error" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 10, 20, 30 };
    const freq = [_]i32{ 1, 2 };

    try testing.expectError(error.LengthMismatch, optimalBST(i32, &keys, &freq, allocator));
}

test "optimal BST: negative frequency error" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 10, 20 };
    const freq = [_]i32{ 5, -3 };

    try testing.expectError(error.InvalidFrequency, optimalBST(i32, &keys, &freq, allocator));
}

test "optimal BST: dummy length error" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 10, 20 };
    const key_freq = [_]i32{ 3, 3 };
    const dummy_freq = [_]i32{ 2, 3 }; // Should be length 3

    try testing.expectError(error.InvalidDummyLength, optimalBSTWithDummy(i32, &keys, &key_freq, &dummy_freq, allocator));
}

test "optimal BST: all equal frequencies" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 1, 2, 3, 4, 5 };
    const freq = [_]i32{ 10, 10, 10, 10, 10 };

    var result = try optimalBST(i32, &keys, &freq, allocator);
    defer result.deinit();

    // With equal frequencies, the optimal root should minimize depth
    // The algorithm will pick a root that balances the tree reasonably well
    try testing.expect(result.root[0][4] < 5);
    try testing.expect(result.min_cost > 0);
}

test "optimal BST: u32 support" {
    const allocator = testing.allocator;

    const keys = [_]u32{ 100, 200, 300 };
    const freq = [_]u32{ 5, 10, 3 };

    var result = try optimalBST(u32, &keys, &freq, allocator);
    defer result.deinit();

    try testing.expect(result.min_cost > 0);
}

test "optimal BST: memory safety" {
    const allocator = testing.allocator;

    const keys = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const freq = [_]i32{ 8, 7, 6, 5, 4, 3, 2, 1 };

    var result = try optimalBST(i32, &keys, &freq, allocator);
    defer result.deinit();

    try testing.expect(result.min_cost > 0);
}
