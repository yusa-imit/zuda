const std = @import("std");
const Allocator = std.mem.Allocator;

/// SparseTable provides O(1) range queries for idempotent operations (min, max, GCD, etc.)
/// after O(n log n) preprocessing time and space.
///
/// An operation f is idempotent if f(x, x) = x for all x.
/// Valid operations: min, max, GCD, bitwise AND/OR
/// Invalid operations: sum, product (not idempotent)
///
/// Memory: O(n log n)
/// Build: O(n log n)
/// Query: O(1)
///
/// Type Parameters:
///   T: Element type
///   combineFn: Comptime idempotent binary operation (a, b -> result)
///
pub fn SparseTable(
    comptime T: type,
    comptime combineFn: fn (a: T, b: T) T,
) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        data: []const T, // Original data (borrowed, not owned)
        table: [][]T, // table[k][i] = combine(data[i..i+2^k])
        log_table: []u32, // Precomputed floor(log2(i))
        n: usize,
        max_log: u32,

        /// Initialize a SparseTable from a slice of data.
        /// The input slice must remain valid for the lifetime of the SparseTable.
        ///
        /// Time: O(n log n)
        /// Space: O(n log n)
        pub fn init(allocator: Allocator, data: []const T) !Self {
            if (data.len == 0) {
                return error.EmptyData;
            }

            const n = data.len;
            const max_log = if (n > 1) std.math.log2_int(usize, n) + 1 else 1;

            // Allocate table[0..max_log][0..n]
            const table = try allocator.alloc([]T, max_log);
            errdefer allocator.free(table);

            for (table, 0..) |*row, k| {
                const len = if (k == 0) n else n -| ((@as(usize, 1) << @intCast(k)) - 1);
                row.* = try allocator.alloc(T, len);
            }
            errdefer {
                for (table) |row| {
                    allocator.free(row);
                }
            }

            // Build log table
            const log_table = try allocator.alloc(u32, n + 1);
            errdefer allocator.free(log_table);

            log_table[0] = 0;
            log_table[1] = 0;
            for (2..n + 1) |i| {
                log_table[i] = log_table[i / 2] + 1;
            }

            // Initialize first row (intervals of length 1)
            @memcpy(table[0][0..n], data);

            // Build sparse table using dynamic programming
            // table[k][i] = combine(table[k-1][i], table[k-1][i + 2^(k-1)])
            for (1..max_log) |k| {
                const k_u5: u5 = @intCast(k);
                const step: usize = @as(usize, 1) << @intCast(k - 1);
                const len = n -| ((@as(usize, 1) << k_u5) - 1);

                for (0..len) |i| {
                    const left = table[k - 1][i];
                    const right = table[k - 1][i + step];
                    table[k][i] = combineFn(left, right);
                }
            }

            return Self{
                .allocator = allocator,
                .data = data,
                .table = table,
                .log_table = log_table,
                .n = n,
                .max_log = @intCast(max_log),
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: *Self) void {
            for (self.table) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(self.table);
            self.allocator.free(self.log_table);
            self.* = undefined;
        }

        /// Query the result of combining all elements in the range [left, right).
        /// Returns error.InvalidRange if left >= right or indices are out of bounds.
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn query(self: *const Self, left: usize, right: usize) !T {
            if (left >= right) return error.InvalidRange;
            if (right > self.n) return error.OutOfBounds;

            const len = right - left;
            const k = self.log_table[len];
            const k_usize: usize = @intCast(k);
            const step: usize = @as(usize, 1) << @intCast(k);

            // Combine two overlapping intervals of length 2^k
            // [left, left + 2^k) and [right - 2^k, right)
            const left_val = self.table[k_usize][left];
            const right_val = self.table[k_usize][right - step];

            return combineFn(left_val, right_val);
        }

        /// Get the number of elements in the sparse table.
        pub fn count(self: *const Self) usize {
            return self.n;
        }

        /// Check if the sparse table is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.n == 0;
        }

        /// Validate internal invariants (for debugging).
        pub fn validate(self: *const Self) !void {
            if (self.n == 0) return error.EmptyTable;
            if (self.table.len != self.max_log) return error.InvalidTableSize;

            // Verify first row matches original data
            for (self.data, 0..) |val, i| {
                if (self.table[0][i] != val) return error.InvalidFirstRow;
            }

            // Verify table dimensions
            for (self.table, 0..) |row, k| {
                const k_u5: u5 = @intCast(k);
                const expected_len = self.n -| ((@as(usize, 1) << k_u5) - 1);
                if (row.len != expected_len) return error.InvalidRowSize;
            }
        }

        /// Format the sparse table for debugging.
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("SparseTable(n={}, max_log={})", .{ self.n, self.max_log });
        }
    };
}

// ============================================================================
// Common SparseTable Instances
// ============================================================================

/// Min operation (idempotent)
pub fn min(comptime T: type) fn (T, T) T {
    return struct {
        fn f(a: T, b: T) T {
            return @min(a, b);
        }
    }.f;
}

/// Max operation (idempotent)
pub fn max(comptime T: type) fn (T, T) T {
    return struct {
        fn f(a: T, b: T) T {
            return @max(a, b);
        }
    }.f;
}

/// GCD operation (idempotent)
pub fn gcd(comptime T: type) fn (T, T) T {
    return struct {
        fn f(a: T, b: T) T {
            var x = a;
            var y = b;
            while (y != 0) {
                const temp = y;
                y = @mod(x, y);
                x = temp;
            }
            return x;
        }
    }.f;
}

/// Bitwise AND operation (idempotent)
pub fn bitwiseAnd(comptime T: type) fn (T, T) T {
    return struct {
        fn f(a: T, b: T) T {
            return a & b;
        }
    }.f;
}

/// Bitwise OR operation (idempotent)
pub fn bitwiseOr(comptime T: type) fn (T, T) T {
    return struct {
        fn f(a: T, b: T) T {
            return a | b;
        }
    }.f;
}

// ============================================================================
// Tests
// ============================================================================

test "SparseTable: basic min queries" {
    const data = [_]i32{ 1, 3, 2, 7, 9, 11, 3, 5 };
    const ST = SparseTable(i32, min(i32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(usize, 8), st.count());
    try std.testing.expectEqual(false, st.isEmpty());

    // Range minimum queries
    try std.testing.expectEqual(@as(i32, 1), try st.query(0, 4)); // min(1,3,2,7)
    try std.testing.expectEqual(@as(i32, 2), try st.query(1, 4)); // min(3,2,7)
    try std.testing.expectEqual(@as(i32, 3), try st.query(4, 8)); // min(9,11,3,5)
    try std.testing.expectEqual(@as(i32, 1), try st.query(0, 8)); // min(all)
    try std.testing.expectEqual(@as(i32, 3), try st.query(1, 2)); // single element
}

test "SparseTable: basic max queries" {
    const data = [_]i32{ 1, 3, 2, 7, 9, 11, 3, 5 };
    const ST = SparseTable(i32, max(i32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    // Range maximum queries
    try std.testing.expectEqual(@as(i32, 7), try st.query(0, 4)); // max(1,3,2,7)
    try std.testing.expectEqual(@as(i32, 7), try st.query(1, 4)); // max(3,2,7)
    try std.testing.expectEqual(@as(i32, 11), try st.query(4, 8)); // max(9,11,3,5)
    try std.testing.expectEqual(@as(i32, 11), try st.query(0, 8)); // max(all)
}

test "SparseTable: GCD queries" {
    const data = [_]u32{ 12, 18, 24, 30, 36 };
    const ST = SparseTable(u32, gcd(u32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(u32, 6), try st.query(0, 2)); // gcd(12,18) = 6
    try std.testing.expectEqual(@as(u32, 6), try st.query(0, 5)); // gcd(all) = 6
    try std.testing.expectEqual(@as(u32, 6), try st.query(2, 4)); // gcd(24,30) = 6
}

test "SparseTable: bitwise AND queries" {
    const data = [_]u8{ 0b1111, 0b1110, 0b1100, 0b1000 };
    const ST = SparseTable(u8, bitwiseAnd(u8));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(u8, 0b1110), try st.query(0, 2)); // 1111 & 1110
    try std.testing.expectEqual(@as(u8, 0b1000), try st.query(0, 4)); // all
    try std.testing.expectEqual(@as(u8, 0b1000), try st.query(1, 4)); // 1110 & 1100 & 1000
}

test "SparseTable: bitwise OR queries" {
    const data = [_]u8{ 0b0001, 0b0010, 0b0100, 0b1000 };
    const ST = SparseTable(u8, bitwiseOr(u8));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(u8, 0b0011), try st.query(0, 2)); // 0001 | 0010
    try std.testing.expectEqual(@as(u8, 0b1111), try st.query(0, 4)); // all
    try std.testing.expectEqual(@as(u8, 0b1110), try st.query(1, 4)); // 0010 | 0100 | 1000
}

test "SparseTable: single element range" {
    const data = [_]i32{ 5, 10, 15, 20 };
    const ST = SparseTable(i32, min(i32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    // Single element queries should return the element itself
    try std.testing.expectEqual(@as(i32, 5), try st.query(0, 1));
    try std.testing.expectEqual(@as(i32, 10), try st.query(1, 2));
    try std.testing.expectEqual(@as(i32, 15), try st.query(2, 3));
    try std.testing.expectEqual(@as(i32, 20), try st.query(3, 4));
}

test "SparseTable: error cases" {
    const data = [_]i32{ 1, 2, 3 };
    const ST = SparseTable(i32, min(i32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    // Invalid ranges
    try std.testing.expectError(error.InvalidRange, st.query(2, 2)); // empty range
    try std.testing.expectError(error.InvalidRange, st.query(3, 2)); // left > right
    try std.testing.expectError(error.OutOfBounds, st.query(0, 5)); // right out of bounds
    try std.testing.expectError(error.OutOfBounds, st.query(2, 4)); // right out of bounds
}

test "SparseTable: empty data" {
    const data = [_]i32{};
    const ST = SparseTable(i32, min(i32));

    try std.testing.expectError(error.EmptyData, ST.init(std.testing.allocator, &data));
}

test "SparseTable: power of two size" {
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // size = 2^3
    const ST = SparseTable(i32, min(i32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(u32, 4), st.max_log); // log2(8) + 1 = 4

    // Test various ranges
    try std.testing.expectEqual(@as(i32, 1), try st.query(0, 8));
    try std.testing.expectEqual(@as(i32, 1), try st.query(0, 4));
    try std.testing.expectEqual(@as(i32, 5), try st.query(4, 8));
    try std.testing.expectEqual(@as(i32, 2), try st.query(1, 5));
}

test "SparseTable: non-power of two size" {
    const data = [_]i32{ 5, 1, 3, 9, 2, 7 }; // size = 6
    const ST = SparseTable(i32, max(i32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(u32, 3), st.max_log); // log2(6) + 1 = 3

    // Test various ranges
    try std.testing.expectEqual(@as(i32, 9), try st.query(0, 6)); // max(all)
    try std.testing.expectEqual(@as(i32, 9), try st.query(2, 5)); // max(3,9,2)
    try std.testing.expectEqual(@as(i32, 7), try st.query(5, 6)); // max(7)
}

test "SparseTable: large dataset min" {
    const n = 1000;
    var data: [n]i32 = undefined;
    for (&data, 0..) |*val, i| {
        val.* = @intCast((i * 7 + 13) % 1000); // pseudo-random
    }

    const ST = SparseTable(i32, min(i32));
    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    // Verify against naive min
    var expected: i32 = data[100];
    for (data[100..300]) |val| {
        expected = @min(expected, val);
    }
    try std.testing.expectEqual(expected, try st.query(100, 300));

    // Full range
    expected = data[0];
    for (data) |val| {
        expected = @min(expected, val);
    }
    try std.testing.expectEqual(expected, try st.query(0, n));
}

test "SparseTable: validate invariants" {
    const data = [_]i32{ 1, 2, 3, 4, 5 };
    const ST = SparseTable(i32, min(i32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    try st.validate();

    // Corrupt first row to test validation
    st.table[0][0] = 999;
    try std.testing.expectError(error.InvalidFirstRow, st.validate());
}

test "SparseTable: custom combine function" {
    // Custom function: return larger absolute value
    const absMax = struct {
        fn f(a: i32, b: i32) i32 {
            const abs_a = if (a < 0) -a else a;
            const abs_b = if (b < 0) -b else b;
            return if (abs_a > abs_b) a else b;
        }
    }.f;

    const data = [_]i32{ -10, 5, -3, 8, -15, 2 };
    const ST = SparseTable(i32, absMax);

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(i32, -15), try st.query(0, 6)); // max abs value
    try std.testing.expectEqual(@as(i32, -10), try st.query(0, 3)); // -10 has abs 10
    try std.testing.expectEqual(@as(i32, 8), try st.query(2, 4)); // 8 > |-3|
}

test "SparseTable: idempotence property verification" {
    // Verify that overlapping query segments produce correct results
    // This tests the idempotence property: f(x,x) = x
    const data = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const ST = SparseTable(i32, min(i32));

    var st = try ST.init(std.testing.allocator, &data);
    defer st.deinit();

    // Query ranges where sparse table uses overlapping intervals
    try std.testing.expectEqual(@as(i32, 1), try st.query(0, 5)); // should use overlapping
    try std.testing.expectEqual(@as(i32, 1), try st.query(1, 6)); // should use overlapping
    try std.testing.expectEqual(@as(i32, 1), try st.query(2, 7)); // should use overlapping

    // Verify with naive approach
    var expected: i32 = data[2];
    for (data[2..7]) |val| {
        expected = @min(expected, val);
    }
    try std.testing.expectEqual(expected, try st.query(2, 7));
}
