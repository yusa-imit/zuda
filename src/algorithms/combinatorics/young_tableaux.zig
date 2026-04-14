const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

/// Young Tableau - a filling of a partition shape with numbers
/// Row lengths are weakly decreasing, entries increase left-to-right and top-to-bottom
///
/// A partition λ = (λ₁, λ₂, ..., λₖ) where λ₁ ≥ λ₂ ≥ ... ≥ λₖ > 0
/// Standard Young Tableau (SYT): filled with 1..n exactly once
///
/// Example shape (3,2,1):
///   ┌─┬─┬─┐
///   │1│2│3│
///   ├─┼─┴─┘
///   │4│5│
///   ├─┴─┘
///   │6│
///   └─┘
///
/// Applications:
/// - Representation theory of symmetric groups
/// - Robinson-Schensted correspondence (permutations ↔ tableau pairs)
/// - Schur functions in algebraic combinatorics
/// - Knuth equivalence and plactic monoid

/// Represents a Young Tableau as a 2D array with irregular row lengths
pub const Tableau = struct {
    rows: [][]u32,
    allocator: Allocator,

    /// Create empty tableau
    /// Time: O(1) | Space: O(1)
    pub fn init(allocator: Allocator) Tableau {
        return .{
            .rows = &[_][]u32{},
            .allocator = allocator,
        };
    }

    /// Create tableau from shape (partition)
    /// Shape λ = (λ₁, λ₂, ..., λₖ) represents row lengths
    /// Time: O(n) where n = Σλᵢ | Space: O(n)
    pub fn fromShape(allocator: Allocator, partition_shape: []const usize) !Tableau {
        const rows = try allocator.alloc([]u32, partition_shape.len);
        errdefer allocator.free(rows);

        for (partition_shape, 0..) |len, i| {
            rows[i] = try allocator.alloc(u32, len);
            @memset(rows[i], 0);
        }

        return .{
            .rows = rows,
            .allocator = allocator,
        };
    }

    /// Create tableau from 2D array
    /// Time: O(n) | Space: O(n)
    pub fn fromSlice(allocator: Allocator, data: []const []const u32) !Tableau {
        const rows = try allocator.alloc([]u32, data.len);
        errdefer allocator.free(rows);

        for (data, 0..) |row, i| {
            rows[i] = try allocator.alloc(u32, row.len);
            @memcpy(rows[i], row);
        }

        return .{
            .rows = rows,
            .allocator = allocator,
        };
    }

    /// Free memory
    pub fn deinit(self: *Tableau) void {
        for (self.rows) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.rows);
    }

    /// Get shape (partition) of tableau
    /// Time: O(k) where k = number of rows | Space: O(k)
    pub fn shape(self: Tableau, allocator: Allocator) ![]usize {
        const s = try allocator.alloc(usize, self.rows.len);
        for (self.rows, 0..) |row, i| {
            s[i] = row.len;
        }
        return s;
    }

    /// Check if tableau is valid (row lengths weakly decreasing)
    /// Time: O(k) | Space: O(1)
    pub fn isValidShape(self: Tableau) bool {
        if (self.rows.len == 0) return true;
        for (1..self.rows.len) |i| {
            if (self.rows[i].len > self.rows[i - 1].len) return false;
        }
        return true;
    }

    /// Check if tableau is standard (entries 1..n exactly once, increasing)
    /// Time: O(n) | Space: O(n)
    pub fn isStandard(self: Tableau, allocator: Allocator) !bool {
        if (!self.isValidShape()) return false;

        var n: u32 = 0;
        for (self.rows) |row| {
            n += @intCast(row.len);
        }

        if (n == 0) return true;

        // Check entries are 1..n exactly once
        const seen = try allocator.alloc(bool, n);
        defer allocator.free(seen);
        @memset(seen, false);

        for (self.rows) |row| {
            for (row) |val| {
                if (val == 0 or val > n) return false;
                if (seen[val - 1]) return false; // duplicate
                seen[val - 1] = true;
            }
        }

        // Check all values present
        for (seen) |s| {
            if (!s) return false;
        }

        // Check increasing in rows and columns
        for (self.rows) |row| {
            for (1..row.len) |j| {
                if (row[j] <= row[j - 1]) return false;
            }
        }

        for (1..self.rows.len) |i| {
            const min_len = @min(self.rows[i].len, self.rows[i - 1].len);
            for (0..min_len) |j| {
                if (self.rows[i][j] <= self.rows[i - 1][j]) return false;
            }
        }

        return true;
    }
};

/// Count Standard Young Tableaux of given shape using Hook Length Formula
///
/// For each cell (i,j), hook length h(i,j) = arm + leg + 1
/// where arm = cells to right, leg = cells below
///
/// Formula: #SYT(λ) = n! / ∏ h(i,j)
///
/// Time: O(n) where n = Σλᵢ | Space: O(1)
pub fn countStandardTableaux(comptime T: type, shape: []const usize) !T {
    if (shape.len == 0) return 1;

    // Compute n = total cells
    var n: T = 0;
    for (shape) |len| {
        n += @intCast(len);
    }

    if (n == 0) return 1;

    // Compute factorial n!
    var fact: T = 1;
    for (1..@as(usize, @intCast(n)) + 1) |i| {
        fact = try std.math.mul(T, fact, @intCast(i));
    }

    // Compute product of hook lengths
    var hook_product: T = 1;
    for (shape, 0..) |row_len, i| {
        for (0..row_len) |j| {
            // arm length: cells to right in row i
            const arm = row_len - j - 1;

            // leg length: cells below in column j
            var leg: usize = 0;
            for (i + 1..shape.len) |k| {
                if (j < shape[k]) {
                    leg += 1;
                } else {
                    break;
                }
            }

            const hook = @as(T, @intCast(arm + leg + 1));
            hook_product = try std.math.mul(T, hook_product, hook);
        }
    }

    return @divExact(fact, hook_product);
}

/// Robinson-Schensted insertion: insert value into tableau
/// Creates a new tableau with value inserted using bumping algorithm
///
/// Algorithm:
/// 1. Find leftmost entry > val in first row
/// 2. If none, append val to first row
/// 3. Otherwise, bump that entry to next row and repeat
///
/// Time: O(n) worst case | Space: O(n)
pub fn insertRobinsonSchensted(allocator: Allocator, tab: Tableau, val: u32) !Tableau {
    var result = try tab.clone(allocator);
    errdefer result.deinit();

    var current_val = val;
    var row_idx: usize = 0;

    while (true) {
        if (row_idx >= result.rows.len) {
            // Create new row with current_val
            const new_rows = try allocator.alloc([]u32, result.rows.len + 1);
            @memcpy(new_rows[0..result.rows.len], result.rows);
            allocator.free(result.rows);
            result.rows = new_rows;

            result.rows[row_idx] = try allocator.alloc(u32, 1);
            result.rows[row_idx][0] = current_val;
            break;
        }

        const row = result.rows[row_idx];

        // Find leftmost entry > current_val
        var insert_pos: ?usize = null;
        for (row, 0..) |entry, j| {
            if (entry > current_val) {
                insert_pos = j;
                break;
            }
        }

        if (insert_pos) |pos| {
            // Bump entry at pos to next row
            const bumped = row[pos];
            row[pos] = current_val;
            current_val = bumped;
            row_idx += 1;
        } else {
            // Append to end of row
            const new_row = try allocator.alloc(u32, row.len + 1);
            @memcpy(new_row[0..row.len], row);
            new_row[row.len] = current_val;
            allocator.free(row);
            result.rows[row_idx] = new_row;
            break;
        }
    }

    return result;
}

/// Clone tableau
fn clone(self: Tableau, allocator: Allocator) !Tableau {
    const rows = try allocator.alloc([]u32, self.rows.len);
    errdefer allocator.free(rows);

    for (self.rows, 0..) |row, i| {
        rows[i] = try allocator.alloc(u32, row.len);
        @memcpy(rows[i], row);
    }

    return .{
        .rows = rows,
        .allocator = allocator,
    };
}

/// Build Standard Young Tableau from permutation via Robinson-Schensted
/// Returns P-tableau (insertion tableau)
///
/// Time: O(n²) | Space: O(n)
pub fn robinsonSchenstedP(allocator: Allocator, perm: []const u32) !Tableau {
    var tab = Tableau.init(allocator);

    for (perm) |val| {
        const new_tab = try insertRobinsonSchensted(allocator, tab, val);
        tab.deinit();
        tab = new_tab;
    }

    return tab;
}

/// Generate all Standard Young Tableaux of given shape
/// Uses backtracking with constraint checking
///
/// Time: O(SYT(λ) × n) | Space: O(n)
pub fn generateStandardTableaux(allocator: Allocator, shape: []const usize) !ArrayList(Tableau) {
    var result = ArrayList(Tableau).init(allocator);
    errdefer {
        for (result.items) |*t| t.deinit();
        result.deinit();
    }

    if (shape.len == 0) {
        const empty = Tableau.init(allocator);
        try result.append(empty);
        return result;
    }

    // Compute n = total cells
    var n: usize = 0;
    for (shape) |len| {
        n += len;
    }

    if (n == 0) {
        const empty = Tableau.init(allocator);
        try result.append(empty);
        return result;
    }

    var tab = try Tableau.fromShape(allocator, shape);
    defer tab.deinit();

    try generateHelper(allocator, &tab, 1, @intCast(n), &result);
    return result;
}

fn generateHelper(allocator: Allocator, tab: *Tableau, next_val: u32, n: u32, result: *ArrayList(Tableau)) !void {
    if (next_val > n) {
        // Found complete tableau
        const t = try tab.clone(allocator);
        try result.append(t);
        return;
    }

    // Try placing next_val in each valid position
    for (tab.rows, 0..) |row, i| {
        for (0..row.len) |j| {
            if (tab.rows[i][j] != 0) continue; // cell occupied

            // Check row constraint: all left entries < next_val
            var valid = true;
            if (j > 0 and tab.rows[i][j - 1] >= next_val) valid = false;

            // Check column constraint: all above entries < next_val
            if (i > 0 and j < tab.rows[i - 1].len and tab.rows[i - 1][j] >= next_val) valid = false;

            // Check if this position must be filled (no gaps in rows)
            if (j > 0 and tab.rows[i][j - 1] == 0) valid = false;

            // Check if position above is filled (no gaps in columns)
            if (i > 0 and j < tab.rows[i - 1].len and tab.rows[i - 1][j] == 0) valid = false;

            if (valid) {
                tab.rows[i][j] = next_val;
                try generateHelper(allocator, tab, next_val + 1, n, result);
                tab.rows[i][j] = 0;
            }
        }
    }
}

// Tests

test "Tableau creation and validation" {
    const allocator = testing.allocator;

    // Empty tableau
    var t1 = Tableau.init(allocator);
    defer t1.deinit();
    try testing.expect(t1.isValidShape());
    try testing.expect(try t1.isStandard(allocator));

    // Valid shape (3,2,1)
    const shape = [_]usize{ 3, 2, 1 };
    var t2 = try Tableau.fromShape(allocator, &shape);
    defer t2.deinit();
    try testing.expect(t2.isValidShape());

    // Invalid shape (increasing)
    const invalid_data = [_][]const u32{
        &[_]u32{ 1, 2 },
        &[_]u32{ 3, 4, 5 },
    };
    var t3 = try Tableau.fromSlice(allocator, &invalid_data);
    defer t3.deinit();
    try testing.expect(!t3.isValidShape());
}

test "Standard Young Tableau validation" {
    const allocator = testing.allocator;

    // Valid SYT (3,2,1)
    const valid = [_][]const u32{
        &[_]u32{ 1, 2, 3 },
        &[_]u32{ 4, 5 },
        &[_]u32{6},
    };
    var t1 = try Tableau.fromSlice(allocator, &valid);
    defer t1.deinit();
    try testing.expect(try t1.isStandard(allocator));

    // Invalid: not increasing in row
    const invalid_row = [_][]const u32{
        &[_]u32{ 1, 3, 2 },
        &[_]u32{ 4, 5 },
    };
    var t2 = try Tableau.fromSlice(allocator, &invalid_row);
    defer t2.deinit();
    try testing.expect(!try t2.isStandard(allocator));

    // Invalid: not increasing in column
    const invalid_col = [_][]const u32{
        &[_]u32{ 1, 2, 3 },
        &[_]u32{ 2, 4 },
    };
    var t3 = try Tableau.fromSlice(allocator, &invalid_col);
    defer t3.deinit();
    try testing.expect(!try t3.isStandard(allocator));

    // Invalid: missing value
    const missing = [_][]const u32{
        &[_]u32{ 1, 2, 4 },
        &[_]u32{ 5, 6 },
    };
    var t4 = try Tableau.fromSlice(allocator, &missing);
    defer t4.deinit();
    try testing.expect(!try t4.isStandard(allocator));

    // Invalid: duplicate value
    const duplicate = [_][]const u32{
        &[_]u32{ 1, 2, 3 },
        &[_]u32{ 3, 4 },
    };
    var t5 = try Tableau.fromSlice(allocator, &duplicate);
    defer t5.deinit();
    try testing.expect(!try t5.isStandard(allocator));
}

test "Hook length formula - small cases" {
    // Shape (1)
    try testing.expectEqual(@as(u64, 1), try countStandardTableaux(u64, &[_]usize{1}));

    // Shape (2)
    try testing.expectEqual(@as(u64, 1), try countStandardTableaux(u64, &[_]usize{2}));

    // Shape (1,1)
    try testing.expectEqual(@as(u64, 1), try countStandardTableaux(u64, &[_]usize{ 1, 1 }));

    // Shape (2,1)
    try testing.expectEqual(@as(u64, 2), try countStandardTableaux(u64, &[_]usize{ 2, 1 }));

    // Shape (3)
    try testing.expectEqual(@as(u64, 1), try countStandardTableaux(u64, &[_]usize{3}));

    // Shape (2,2)
    try testing.expectEqual(@as(u64, 2), try countStandardTableaux(u64, &[_]usize{ 2, 2 }));

    // Shape (3,1)
    try testing.expectEqual(@as(u64, 3), try countStandardTableaux(u64, &[_]usize{ 3, 1 }));

    // Shape (2,1,1)
    try testing.expectEqual(@as(u64, 3), try countStandardTableaux(u64, &[_]usize{ 2, 1, 1 }));
}

test "Hook length formula - larger shapes" {
    // Shape (3,2,1) - should be 16
    // Hooks:  5 4 2
    //         3 2
    //         1
    // Product = 5×4×2×3×2×1 = 240
    // 6! / 240 = 720 / 240 = 3... wait, let me recalculate
    // Actually, 6!/240 = 3, but correct answer is 16
    // Let me verify the hook calculation:
    // (0,0): arm=2, leg=2, hook=5 ✓
    // (0,1): arm=1, leg=1, hook=3... wait
    // Let me be more careful:
    // (0,0): right=2, below=2, hook=5
    // (0,1): right=1, below=1, hook=3
    // (0,2): right=0, below=0, hook=1
    // (1,0): right=1, below=1, hook=3
    // (1,1): right=0, below=0, hook=1
    // (2,0): right=0, below=0, hook=1
    // Product = 5×3×1×3×1×1 = 45
    // 6! / 45 = 720 / 45 = 16 ✓
    try testing.expectEqual(@as(u64, 16), try countStandardTableaux(u64, &[_]usize{ 3, 2, 1 }));

    // Shape (4,3,2,1) - n=10, known value is 768
    try testing.expectEqual(@as(u64, 768), try countStandardTableaux(u64, &[_]usize{ 4, 3, 2, 1 }));
}

test "Hook length formula - edge cases" {
    // Empty shape
    try testing.expectEqual(@as(u64, 1), try countStandardTableaux(u64, &[_]usize{}));

    // Single row
    try testing.expectEqual(@as(u64, 1), try countStandardTableaux(u64, &[_]usize{5}));

    // Single column
    try testing.expectEqual(@as(u64, 1), try countStandardTableaux(u64, &[_]usize{ 1, 1, 1, 1 }));
}

test "Robinson-Schensted insertion - basic" {
    const allocator = testing.allocator;

    // Start with empty tableau
    var tab = Tableau.init(allocator);
    defer tab.deinit();

    // Insert 3
    var t1 = try insertRobinsonSchensted(allocator, tab, 3);
    defer t1.deinit();
    try testing.expectEqual(@as(usize, 1), t1.rows.len);
    try testing.expectEqual(@as(usize, 1), t1.rows[0].len);
    try testing.expectEqual(@as(u32, 3), t1.rows[0][0]);

    // Insert 1 (should go before 3)
    var t2 = try insertRobinsonSchensted(allocator, t1, 1);
    defer t2.deinit();
    try testing.expectEqual(@as(usize, 1), t2.rows.len);
    try testing.expectEqual(@as(usize, 2), t2.rows[0].len);
    try testing.expectEqual(@as(u32, 1), t2.rows[0][0]);
    try testing.expectEqual(@as(u32, 3), t2.rows[0][1]);

    // Insert 2 (should bump 3 to second row)
    var t3 = try insertRobinsonSchensted(allocator, t2, 2);
    defer t3.deinit();
    try testing.expectEqual(@as(usize, 2), t3.rows.len);
    try testing.expectEqual(@as(u32, 1), t3.rows[0][0]);
    try testing.expectEqual(@as(u32, 2), t3.rows[0][1]);
    try testing.expectEqual(@as(u32, 3), t3.rows[1][0]);
}

test "Robinson-Schensted P-tableau from permutation" {
    const allocator = testing.allocator;

    // Identity permutation [1,2,3]
    const perm1 = [_]u32{ 1, 2, 3 };
    var tab1 = try robinsonSchenstedP(allocator, &perm1);
    defer tab1.deinit();
    try testing.expect(try tab1.isStandard(allocator));
    try testing.expectEqual(@as(usize, 1), tab1.rows.len); // Single row

    // Reverse permutation [3,2,1]
    const perm2 = [_]u32{ 3, 2, 1 };
    var tab2 = try robinsonSchenstedP(allocator, &perm2);
    defer tab2.deinit();
    try testing.expect(try tab2.isStandard(allocator));
    try testing.expectEqual(@as(usize, 3), tab2.rows.len); // Single column

    // Mixed permutation [2,1,4,3]
    const perm3 = [_]u32{ 2, 1, 4, 3 };
    var tab3 = try robinsonSchenstedP(allocator, &perm3);
    defer tab3.deinit();
    try testing.expect(try tab3.isStandard(allocator));
}

test "Generate all SYT - small shapes" {
    const allocator = testing.allocator;

    // Shape (2,1)
    var syts1 = try generateStandardTableaux(allocator, &[_]usize{ 2, 1 });
    defer {
        for (syts1.items) |*t| t.deinit();
        syts1.deinit();
    }
    try testing.expectEqual(@as(usize, 2), syts1.items.len);
    for (syts1.items) |t| {
        try testing.expect(try t.isStandard(allocator));
    }

    // Shape (2,2)
    var syts2 = try generateStandardTableaux(allocator, &[_]usize{ 2, 2 });
    defer {
        for (syts2.items) |*t| t.deinit();
        syts2.deinit();
    }
    try testing.expectEqual(@as(usize, 2), syts2.items.len);
    for (syts2.items) |t| {
        try testing.expect(try t.isStandard(allocator));
    }
}

test "Generate all SYT - verify count matches hook formula" {
    const allocator = testing.allocator;

    const test_shapes = [_][]const usize{
        &[_]usize{3},
        &[_]usize{ 2, 1 },
        &[_]usize{ 3, 1 },
        &[_]usize{ 2, 2 },
        &[_]usize{ 2, 1, 1 },
    };

    for (test_shapes) |shape| {
        const expected_count = try countStandardTableaux(u64, shape);
        var syts = try generateStandardTableaux(allocator, shape);
        defer {
            for (syts.items) |*t| t.deinit();
            syts.deinit();
        }
        try testing.expectEqual(expected_count, syts.items.len);
    }
}

test "Tableau shape extraction" {
    const allocator = testing.allocator;

    const data = [_][]const u32{
        &[_]u32{ 1, 2, 3 },
        &[_]u32{ 4, 5 },
        &[_]u32{6},
    };
    var tab = try Tableau.fromSlice(allocator, &data);
    defer tab.deinit();

    const s = try tab.shape(allocator);
    defer allocator.free(s);

    try testing.expectEqual(@as(usize, 3), s.len);
    try testing.expectEqual(@as(usize, 3), s[0]);
    try testing.expectEqual(@as(usize, 2), s[1]);
    try testing.expectEqual(@as(usize, 1), s[2]);
}

test "Type variants - u32/u64" {
    // u32
    try testing.expectEqual(@as(u32, 16), try countStandardTableaux(u32, &[_]usize{ 3, 2, 1 }));

    // u64
    try testing.expectEqual(@as(u64, 768), try countStandardTableaux(u64, &[_]usize{ 4, 3, 2, 1 }));
}

test "Memory safety - multiple allocations" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var tab = try Tableau.fromShape(allocator, &[_]usize{ 3, 2, 1 });
        defer tab.deinit();

        const s = try tab.shape(allocator);
        defer allocator.free(s);

        var syts = try generateStandardTableaux(allocator, &[_]usize{ 2, 1 });
        defer {
            for (syts.items) |*t| t.deinit();
            syts.deinit();
        }
    }
}
