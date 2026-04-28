//! Sparse Matrix Formats and Operations
//!
//! Provides three standard sparse matrix storage formats optimized for different use cases:
//!
//! - **COO (Coordinate)**: Simplest format, stores (row, col, value) triplets
//!   - Best for: matrix construction, format conversion
//!   - Operations: O(nnz) iteration, O(nnz log nnz) sorting
//!
//! - **CSR (Compressed Sparse Row)**: Row-wise compressed storage
//!   - Best for: row-wise operations, matrix-vector multiply
//!   - Operations: O(nnz) row iteration, O(m) row access
//!
//! - **CSC (Compressed Sparse Column)**: Column-wise compressed storage
//!   - Best for: column-wise operations, transpose operations
//!   - Operations: O(nnz) column iteration, O(n) column access
//!
//! Use cases:
//! - Large-scale linear systems (FEM, CFD simulations)
//! - Graph algorithms (adjacency matrices)
//! - Machine learning (sparse feature matrices)
//! - Network analysis
//!
//! All formats support comptime numeric types and explicit allocator passing.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;
const math = std.math;

/// COO (Coordinate) Format
///
/// Stores sparse matrix as triplets of (row, col, value).
/// Memory: 3 × nnz (2 arrays of usize + 1 array of T)
///
/// Advantages:
/// - Simple construction and modification
/// - Easy format conversion
/// - Duplicates allowed (can be summed during conversion)
///
/// Disadvantages:
/// - No efficient row/column access
/// - Not optimized for arithmetic operations
///
/// Time: O(1) | Space: O(nnz)
pub fn COO(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Matrix dimensions
        rows: usize,
        cols: usize,

        /// Row indices (length = nnz)
        row_indices: std.ArrayList(usize),

        /// Column indices (length = nnz)
        col_indices: std.ArrayList(usize),

        /// Non-zero values (length = nnz)
        values: std.ArrayList(T),

        /// Allocator used for dynamic arrays
        allocator: Allocator,

        /// Initialize empty COO matrix
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, rows: usize, cols: usize) Self {
            const ArrayListUsize = std.ArrayList(usize);
            const ArrayListT = std.ArrayList(T);
            return Self{
                .rows = rows,
                .cols = cols,
                .row_indices = ArrayListUsize{},
                .col_indices = ArrayListUsize{},
                .values = ArrayListT{},
                .allocator = allocator,
            };
        }

        /// Initialize COO matrix with preallocated capacity
        ///
        /// Time: O(1) | Space: O(capacity)
        pub fn initCapacity(allocator: Allocator, rows: usize, cols: usize, capacity: usize) !Self {
            const ArrayListUsize = std.ArrayList(usize);
            const ArrayListT = std.ArrayList(T);

            var row_indices = try ArrayListUsize.initCapacity(allocator, capacity);
            errdefer row_indices.deinit(allocator);

            var col_indices = try ArrayListUsize.initCapacity(allocator, capacity);
            errdefer col_indices.deinit(allocator);

            const values = try ArrayListT.initCapacity(allocator, capacity);

            return Self{
                .rows = rows,
                .cols = cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.row_indices.deinit(self.allocator);
            self.col_indices.deinit(self.allocator);
            self.values.deinit(self.allocator);
        }

        /// Add a non-zero element
        ///
        /// Note: Duplicates are allowed and will be summed during conversion to CSR/CSC
        ///
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn append(self: *Self, row: usize, col: usize, value: T) !void {
            if (row >= self.rows or col >= self.cols) {
                return error.OutOfBounds;
            }

            try self.row_indices.append(self.allocator, row);
            try self.col_indices.append(self.allocator, col);
            try self.values.append(self.allocator, value);
        }

        /// Get number of non-zero elements
        ///
        /// Time: O(1) | Space: O(1)
        pub fn nnz(self: *const Self) usize {
            return self.values.items.len;
        }

        /// Check if matrix is empty
        ///
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.values.items.len == 0;
        }

        /// Sort entries by (row, col) for efficient conversion to CSR
        ///
        /// Time: O(nnz log nnz) | Space: O(nnz)
        pub fn sort(self: *Self) !void {
            const Context = struct {
                rows: []const usize,
                cols: []const usize,

                pub fn lessThan(ctx: @This(), a_index: usize, b_index: usize) bool {
                    if (ctx.rows[a_index] != ctx.rows[b_index]) {
                        return ctx.rows[a_index] < ctx.rows[b_index];
                    }
                    return ctx.cols[a_index] < ctx.cols[b_index];
                }
            };

            const n = self.nnz();
            if (n == 0) return;

            // Create index array
            const indices = try self.allocator.alloc(usize, n);
            defer self.allocator.free(indices);

            for (indices, 0..) |*idx, i| {
                idx.* = i;
            }

            // Sort indices based on (row, col)
            const context = Context{
                .rows = self.row_indices.items,
                .cols = self.col_indices.items,
            };

            std.mem.sort(usize, indices, context, Context.lessThan);

            // Reorder arrays
            const temp_rows = try self.allocator.alloc(usize, n);
            defer self.allocator.free(temp_rows);
            const temp_cols = try self.allocator.alloc(usize, n);
            defer self.allocator.free(temp_cols);
            const temp_vals = try self.allocator.alloc(T, n);
            defer self.allocator.free(temp_vals);

            for (indices, 0..) |idx, i| {
                temp_rows[i] = self.row_indices.items[idx];
                temp_cols[i] = self.col_indices.items[idx];
                temp_vals[i] = self.values.items[idx];
            }

            mem.copyForwards(usize, self.row_indices.items, temp_rows);
            mem.copyForwards(usize, self.col_indices.items, temp_cols);
            mem.copyForwards(T, self.values.items, temp_vals);
        }

        /// Validate internal invariants
        ///
        /// Checks:
        /// - All arrays have equal length
        /// - Row indices are within bounds [0, rows)
        /// - Column indices are within bounds [0, cols)
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            const n = self.row_indices.items.len;

            // Check array lengths match
            if (self.col_indices.items.len != n) {
                return error.InvalidArrayLength;
            }
            if (self.values.items.len != n) {
                return error.InvalidArrayLength;
            }

            // Check indices are within bounds
            for (self.row_indices.items) |row| {
                if (row >= self.rows) {
                    return error.IndexOutOfBounds;
                }
            }
            for (self.col_indices.items) |col| {
                if (col >= self.cols) {
                    return error.IndexOutOfBounds;
                }
            }
        }
    };
}

/// CSR (Compressed Sparse Row) Format
///
/// Stores sparse matrix in compressed row format:
/// - row_ptr[i] points to start of row i in col_indices and values
/// - row_ptr[i+1] - row_ptr[i] = number of non-zeros in row i
///
/// Memory: (m+1) + 2×nnz (row pointers + col indices + values)
///
/// Advantages:
/// - Efficient row-wise access O(1)
/// - Fast matrix-vector multiply O(nnz)
/// - Cache-friendly row iteration
///
/// Disadvantages:
/// - Column access is O(nnz)
/// - Modification requires rebuild
///
/// Time: O(1) | Space: O(m + nnz)
pub fn CSR(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Matrix dimensions
        rows: usize,
        cols: usize,

        /// Row pointers (length = rows + 1)
        /// row_ptr[i] = start index of row i in col_indices/values
        /// row_ptr[rows] = nnz (total number of non-zeros)
        row_ptr: []usize,

        /// Column indices (length = nnz)
        col_indices: []usize,

        /// Non-zero values (length = nnz)
        values: []T,

        /// Allocator used for dynamic arrays
        allocator: Allocator,

        /// Initialize CSR matrix from COO format
        ///
        /// Note: Input COO matrix must be sorted by (row, col)
        /// Duplicates will be summed
        ///
        /// Time: O(nnz) | Space: O(m + nnz)
        pub fn fromCOO(allocator: Allocator, coo: *COO(T)) !Self {
            const n = coo.nnz();
            const m = coo.rows;

            // Allocate arrays
            const row_ptr = try allocator.alloc(usize, m + 1);
            errdefer allocator.free(row_ptr);

            const col_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(col_indices);

            const values = try allocator.alloc(T, n);

            // Initialize row_ptr to zero
            for (row_ptr) |*p| {
                p.* = 0;
            }

            if (n == 0) {
                return Self{
                    .rows = m,
                    .cols = coo.cols,
                    .row_ptr = row_ptr,
                    .col_indices = col_indices,
                    .values = values,
                    .allocator = allocator,
                };
            }

            // Count non-zeros per row
            for (coo.row_indices.items) |row| {
                row_ptr[row + 1] += 1;
            }

            // Cumulative sum to get row pointers
            for (1..m + 1) |i| {
                row_ptr[i] += row_ptr[i - 1];
            }

            // Fill col_indices and values
            // Use temporary counters to track insertion positions
            const temp_ptr = try allocator.alloc(usize, m + 1);
            defer allocator.free(temp_ptr);
            mem.copyForwards(usize, temp_ptr, row_ptr);

            for (coo.row_indices.items, 0..) |row, i| {
                const pos = temp_ptr[row];
                col_indices[pos] = coo.col_indices.items[i];
                values[pos] = coo.values.items[i];
                temp_ptr[row] += 1;
            }

            return Self{
                .rows = m,
                .cols = coo.cols,
                .row_ptr = row_ptr,
                .col_indices = col_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.row_ptr);
            self.allocator.free(self.col_indices);
            self.allocator.free(self.values);
        }

        /// Get number of non-zero elements
        ///
        /// Time: O(1) | Space: O(1)
        pub fn nnz(self: *const Self) usize {
            return self.values.len;
        }

        /// Get non-zero count for specific row
        ///
        /// Time: O(1) | Space: O(1)
        pub fn rowNnz(self: *const Self, row: usize) usize {
            if (row >= self.rows) return 0;
            return self.row_ptr[row + 1] - self.row_ptr[row];
        }

        /// Get element at (row, col), returns 0 if not found
        ///
        /// Time: O(nnz_row) | Space: O(1)
        pub fn get(self: *const Self, row: usize, col: usize) T {
            if (row >= self.rows or col >= self.cols) return 0;

            const start = self.row_ptr[row];
            const end = self.row_ptr[row + 1];

            for (self.col_indices[start..end], start..) |c, i| {
                if (c == col) return self.values[i];
            }

            return 0;
        }

        /// Entry returned by row iterator
        /// Transpose CSR matrix to CSC format
        ///
        /// Converts row-wise compressed format to column-wise compressed format.
        /// Effectively computes A^T (transpose).
        ///
        /// Time: O(nnz) | Space: O(n + nnz)
        pub fn transpose(self: *const Self, allocator: Allocator) !CSC(T) {
            const n = self.nnz();
            const col_count = self.cols;

            // Allocate CSC arrays
            const col_ptr = try allocator.alloc(usize, col_count + 1);
            errdefer allocator.free(col_ptr);

            const row_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(row_indices);

            const values = try allocator.alloc(T, n);

            // Initialize col_ptr to zero
            for (col_ptr) |*p| {
                p.* = 0;
            }

            if (n == 0) {
                return CSC(T){
                    .rows = self.cols, // Swap dimensions
                    .cols = self.rows,
                    .col_ptr = col_ptr,
                    .row_indices = row_indices,
                    .values = values,
                    .allocator = allocator,
                };
            }

            // Count non-zeros per column (becomes col_ptr)
            for (self.col_indices) |col| {
                col_ptr[col + 1] += 1;
            }

            // Cumulative sum to get column pointers
            for (1..col_count + 1) |i| {
                col_ptr[i] += col_ptr[i - 1];
            }

            // Fill row_indices and values
            const temp_ptr = try allocator.alloc(usize, col_count + 1);
            defer allocator.free(temp_ptr);
            mem.copyForwards(usize, temp_ptr, col_ptr);

            for (0..self.rows) |row| {
                const start = self.row_ptr[row];
                const end = self.row_ptr[row + 1];
                for (start..end) |idx| {
                    const col = self.col_indices[idx];
                    const pos = temp_ptr[col];
                    row_indices[pos] = row;
                    values[pos] = self.values[idx];
                    temp_ptr[col] += 1;
                }
            }

            return CSC(T){
                .rows = self.cols, // Swap dimensions
                .cols = self.rows,
                .col_ptr = col_ptr,
                .row_indices = row_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Sparse matrix-vector multiplication: y = A*x
        ///
        /// Computes the product of this sparse matrix (m×n) with a dense vector x (n×1),
        /// producing a dense vector y (m×1). Uses row-wise accumulation.
        ///
        /// Time: O(nnz) | Space: O(m) for result vector
        pub fn matvec(self: *const Self, allocator: Allocator, x: []const T) ![]T {
            const y = try allocator.alloc(T, self.rows);
            errdefer allocator.free(y);

            // Initialize result to zero
            for (y) |*val| {
                val.* = 0;
            }

            // Row-wise accumulation
            for (0..self.rows) |row| {
                const start = self.row_ptr[row];
                const end = self.row_ptr[row + 1];
                var sum: T = 0;
                for (start..end) |idx| {
                    const col = self.col_indices[idx];
                    sum += self.values[idx] * x[col];
                }
                y[row] = sum;
            }

            return y;
        }

        pub const Entry = struct {
            col: usize,
            value: T,
        };

        /// Iterator for non-zero elements in a row
        pub const RowIterator = struct {
            col_indices: []const usize,
            values: []const T,
            index: usize,
            end: usize,

            pub fn next(self: *RowIterator) ?Entry {
                if (self.index >= self.end) return null;
                const result = Entry{
                    .col = self.col_indices[self.index],
                    .value = self.values[self.index],
                };
                self.index += 1;
                return result;
            }
        };

        /// Get iterator for row elements
        ///
        /// Time: O(1) | Space: O(1)
        pub fn rowIterator(self: *const Self, row: usize) RowIterator {
            if (row >= self.rows) {
                return RowIterator{
                    .col_indices = &[_]usize{},
                    .values = &[_]T{},
                    .index = 0,
                    .end = 0,
                };
            }

            const start = self.row_ptr[row];
            const end = self.row_ptr[row + 1];

            return RowIterator{
                .col_indices = self.col_indices,
                .values = self.values,
                .index = start,
                .end = end,
            };
        }

        /// Validate internal invariants
        ///
        /// Checks:
        /// - row_ptr has correct length (rows + 1)
        /// - col_indices and values have equal length
        /// - row_ptr is monotonically increasing
        /// - row_ptr[0] == 0
        /// - row_ptr[rows] == nnz
        /// - Column indices are within bounds [0, cols)
        /// - Column indices within each row are sorted (optional, CSR doesn't require this but it's good practice)
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            const n = self.nnz();

            // Check row_ptr length
            if (self.row_ptr.len != self.rows + 1) {
                return error.InvalidRowPtrLength;
            }

            // Check col_indices and values have equal length
            if (self.col_indices.len != n) {
                return error.InvalidArrayLength;
            }
            if (self.values.len != n) {
                return error.InvalidArrayLength;
            }

            // Check row_ptr is monotonically increasing
            if (self.row_ptr[0] != 0) {
                return error.InvalidRowPtrStart;
            }
            for (1..self.row_ptr.len) |i| {
                if (self.row_ptr[i] < self.row_ptr[i - 1]) {
                    return error.RowPtrNotMonotonic;
                }
            }
            if (self.row_ptr[self.rows] != n) {
                return error.InvalidRowPtrEnd;
            }

            // Check column indices are within bounds
            for (self.col_indices) |col| {
                if (col >= self.cols) {
                    return error.IndexOutOfBounds;
                }
            }
        }
    };
}

/// CSC (Compressed Sparse Column) Format
///
/// Stores sparse matrix in compressed column format (transpose of CSR):
/// - col_ptr[j] points to start of column j in row_indices and values
/// - col_ptr[j+1] - col_ptr[j] = number of non-zeros in column j
///
/// Memory: (n+1) + 2×nnz (column pointers + row indices + values)
///
/// Advantages:
/// - Efficient column-wise access O(1)
/// - Fast transpose operations
/// - Natural for column-oriented algorithms
///
/// Disadvantages:
/// - Row access is O(nnz)
/// - Modification requires rebuild
///
/// Time: O(1) | Space: O(n + nnz)
pub fn CSC(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Matrix dimensions
        rows: usize,
        cols: usize,

        /// Column pointers (length = cols + 1)
        col_ptr: []usize,

        /// Row indices (length = nnz)
        row_indices: []usize,

        /// Non-zero values (length = nnz)
        values: []T,

        /// Allocator used for dynamic arrays
        allocator: Allocator,

        /// Initialize CSC matrix from COO format
        ///
        /// Note: Input COO matrix will be sorted by (col, row)
        ///
        /// Time: O(nnz log nnz) | Space: O(n + nnz)
        pub fn fromCOO(allocator: Allocator, coo: *COO(T)) !Self {
            const n = coo.nnz();
            const cols_count = coo.cols;

            // Sort COO by column first
            try sortCOOByColumn(coo);

            // Allocate arrays
            const col_ptr = try allocator.alloc(usize, cols_count + 1);
            errdefer allocator.free(col_ptr);

            const row_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(row_indices);

            const values = try allocator.alloc(T, n);

            // Initialize col_ptr
            for (col_ptr) |*p| {
                p.* = 0;
            }

            if (n == 0) {
                return Self{
                    .rows = coo.rows,
                    .cols = cols_count,
                    .col_ptr = col_ptr,
                    .row_indices = row_indices,
                    .values = values,
                    .allocator = allocator,
                };
            }

            // Count non-zeros per column
            for (coo.col_indices.items) |col| {
                col_ptr[col + 1] += 1;
            }

            // Cumulative sum
            for (1..cols_count + 1) |i| {
                col_ptr[i] += col_ptr[i - 1];
            }

            // Fill row_indices and values
            const temp_ptr = try allocator.alloc(usize, cols_count + 1);
            defer allocator.free(temp_ptr);
            mem.copyForwards(usize, temp_ptr, col_ptr);

            for (coo.col_indices.items, 0..) |col, i| {
                const pos = temp_ptr[col];
                row_indices[pos] = coo.row_indices.items[i];
                values[pos] = coo.values.items[i];
                temp_ptr[col] += 1;
            }

            return Self{
                .rows = coo.rows,
                .cols = cols_count,
                .col_ptr = col_ptr,
                .row_indices = row_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.col_ptr);
            self.allocator.free(self.row_indices);
            self.allocator.free(self.values);
        }

        /// Get number of non-zero elements
        ///
        /// Time: O(1) | Space: O(1)
        pub fn nnz(self: *const Self) usize {
            return self.values.len;
        }

        /// Get non-zero count for specific column
        ///
        /// Time: O(1) | Space: O(1)
        pub fn colNnz(self: *const Self, col: usize) usize {
            if (col >= self.cols) return 0;
            return self.col_ptr[col + 1] - self.col_ptr[col];
        }

        /// Get element at (row, col), returns 0 if not found
        ///
        /// Time: O(nnz_col) | Space: O(1)
        pub fn get(self: *const Self, row: usize, col: usize) T {
            if (row >= self.rows or col >= self.cols) return 0;

            const start = self.col_ptr[col];
            const end = self.col_ptr[col + 1];

            for (self.row_indices[start..end], start..) |r, i| {
                if (r == row) return self.values[i];
            }

            return 0;
        }

        /// Transpose CSC matrix to CSR format
        ///
        /// Converts column-wise compressed format to row-wise compressed format.
        /// Effectively computes A^T (transpose).
        ///
        /// Time: O(nnz) | Space: O(m + nnz)
        pub fn transpose(self: *const Self, allocator: Allocator) !CSR(T) {
            const n = self.nnz();
            const row_count = self.rows;

            // Allocate CSR arrays
            const row_ptr = try allocator.alloc(usize, row_count + 1);
            errdefer allocator.free(row_ptr);

            const col_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(col_indices);

            const values = try allocator.alloc(T, n);

            // Initialize row_ptr to zero
            for (row_ptr) |*p| {
                p.* = 0;
            }

            if (n == 0) {
                return CSR(T){
                    .rows = self.cols, // Swap dimensions
                    .cols = self.rows,
                    .row_ptr = row_ptr,
                    .col_indices = col_indices,
                    .values = values,
                    .allocator = allocator,
                };
            }

            // Count non-zeros per row (becomes row_ptr)
            for (self.row_indices) |row| {
                row_ptr[row + 1] += 1;
            }

            // Cumulative sum to get row pointers
            for (1..row_count + 1) |i| {
                row_ptr[i] += row_ptr[i - 1];
            }

            // Fill col_indices and values
            const temp_ptr = try allocator.alloc(usize, row_count + 1);
            defer allocator.free(temp_ptr);
            mem.copyForwards(usize, temp_ptr, row_ptr);

            for (0..self.cols) |col| {
                const start = self.col_ptr[col];
                const end = self.col_ptr[col + 1];
                for (start..end) |idx| {
                    const row = self.row_indices[idx];
                    const pos = temp_ptr[row];
                    col_indices[pos] = col;
                    values[pos] = self.values[idx];
                    temp_ptr[row] += 1;
                }
            }

            return CSR(T){
                .rows = self.cols, // Swap dimensions
                .cols = self.rows,
                .row_ptr = row_ptr,
                .col_indices = col_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Sparse matrix-vector multiplication: y = A*x
        ///
        /// Computes the product of this sparse matrix (m×n) with a dense vector x (n×1),
        /// producing a dense vector y (m×1). Uses column-wise accumulation.
        ///
        /// Time: O(nnz) | Space: O(m) for result vector
        pub fn matvec(self: *const Self, allocator: Allocator, x: []const T) ![]T {
            const y = try allocator.alloc(T, self.rows);
            errdefer allocator.free(y);

            // Initialize result to zero
            for (y) |*val| {
                val.* = 0;
            }

            // Column-wise accumulation: y += x[j] * A[:,j]
            for (0..self.cols) |col| {
                const start = self.col_ptr[col];
                const end = self.col_ptr[col + 1];
                const x_val = x[col];
                for (start..end) |idx| {
                    const row = self.row_indices[idx];
                    y[row] += self.values[idx] * x_val;
                }
            }

            return y;
        }

        /// Validate internal invariants
        ///
        /// Checks:
        /// - col_ptr has correct length (cols + 1)
        /// - row_indices and values have equal length
        /// - col_ptr is monotonically increasing
        /// - col_ptr[0] == 0
        /// - col_ptr[cols] == nnz
        /// - Row indices are within bounds [0, rows)
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            const n = self.nnz();

            // Check col_ptr length
            if (self.col_ptr.len != self.cols + 1) {
                return error.InvalidColPtrLength;
            }

            // Check row_indices and values have equal length
            if (self.row_indices.len != n) {
                return error.InvalidArrayLength;
            }
            if (self.values.len != n) {
                return error.InvalidArrayLength;
            }

            // Check col_ptr is monotonically increasing
            if (self.col_ptr[0] != 0) {
                return error.InvalidColPtrStart;
            }
            for (1..self.col_ptr.len) |i| {
                if (self.col_ptr[i] < self.col_ptr[i - 1]) {
                    return error.ColPtrNotMonotonic;
                }
            }
            if (self.col_ptr[self.cols] != n) {
                return error.InvalidColPtrEnd;
            }

            // Check row indices are within bounds
            for (self.row_indices) |row| {
                if (row >= self.rows) {
                    return error.IndexOutOfBounds;
                }
            }
        }
    };
}

/// Helper function to sort COO matrix by column then row
fn sortCOOByColumn(coo: anytype) !void {
    const Context = struct {
        cols: []const usize,
        rows: []const usize,

        pub fn lessThan(ctx: @This(), a_index: usize, b_index: usize) bool {
            if (ctx.cols[a_index] != ctx.cols[b_index]) {
                return ctx.cols[a_index] < ctx.cols[b_index];
            }
            return ctx.rows[a_index] < ctx.rows[b_index];
        }
    };

    const n = coo.nnz();
    if (n == 0) return;

    const indices = try coo.allocator.alloc(usize, n);
    defer coo.allocator.free(indices);

    for (indices, 0..) |*idx, i| {
        idx.* = i;
    }

    const context = Context{
        .cols = coo.col_indices.items,
        .rows = coo.row_indices.items,
    };

    std.mem.sort(usize, indices, context, Context.lessThan);

    // Reorder
    const temp_rows = try coo.allocator.alloc(usize, n);
    defer coo.allocator.free(temp_rows);
    const temp_cols = try coo.allocator.alloc(usize, n);
    defer coo.allocator.free(temp_cols);
    const temp_vals = try coo.allocator.alloc(@TypeOf(coo.values.items[0]), n);
    defer coo.allocator.free(temp_vals);

    for (indices, 0..) |idx, i| {
        temp_rows[i] = coo.row_indices.items[idx];
        temp_cols[i] = coo.col_indices.items[idx];
        temp_vals[i] = coo.values.items[idx];
    }

    mem.copyForwards(usize, coo.row_indices.items, temp_rows);
    mem.copyForwards(usize, coo.col_indices.items, temp_cols);
    mem.copyForwards(@TypeOf(coo.values.items[0]), coo.values.items, temp_vals);
}

// ============================================================================
// Tests
// ============================================================================

test "COO: init and basic operations" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try testing.expectEqual(@as(usize, 3), coo.rows);
    try testing.expectEqual(@as(usize, 3), coo.cols);
    try testing.expectEqual(@as(usize, 0), coo.nnz());
    try testing.expect(coo.isEmpty());
}

test "COO: append elements" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);

    try testing.expectEqual(@as(usize, 3), coo.nnz());
    try testing.expect(!coo.isEmpty());

    try testing.expectEqual(@as(usize, 0), coo.row_indices.items[0]);
    try testing.expectEqual(@as(usize, 0), coo.col_indices.items[0]);
    try testing.expectEqual(@as(f64, 1.0), coo.values.items[0]);
}

test "COO: out of bounds" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try testing.expectError(error.OutOfBounds, coo.append(3, 0, 1.0));
    try testing.expectError(error.OutOfBounds, coo.append(0, 3, 1.0));
}

test "COO: sort" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(2, 1, 6.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 0, 4.0);
    try coo.append(0, 0, 1.0);

    try coo.sort();

    // Should be sorted by (row, col)
    try testing.expectEqual(@as(usize, 0), coo.row_indices.items[0]);
    try testing.expectEqual(@as(usize, 0), coo.col_indices.items[0]);
    try testing.expectEqual(@as(f64, 1.0), coo.values.items[0]);

    try testing.expectEqual(@as(usize, 0), coo.row_indices.items[1]);
    try testing.expectEqual(@as(usize, 2), coo.col_indices.items[1]);
    try testing.expectEqual(@as(f64, 3.0), coo.values.items[1]);

    try testing.expectEqual(@as(usize, 1), coo.row_indices.items[2]);
    try testing.expectEqual(@as(usize, 0), coo.col_indices.items[2]);
    try testing.expectEqual(@as(f64, 4.0), coo.values.items[2]);

    try testing.expectEqual(@as(usize, 2), coo.row_indices.items[3]);
    try testing.expectEqual(@as(usize, 1), coo.col_indices.items[3]);
    try testing.expectEqual(@as(f64, 6.0), coo.values.items[3]);
}

test "CSR: fromCOO basic" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(usize, 3), csr.rows);
    try testing.expectEqual(@as(usize, 3), csr.cols);
    try testing.expectEqual(@as(usize, 3), csr.nnz());

    // Check row pointers
    try testing.expectEqual(@as(usize, 0), csr.row_ptr[0]);
    try testing.expectEqual(@as(usize, 1), csr.row_ptr[1]);
    try testing.expectEqual(@as(usize, 2), csr.row_ptr[2]);
    try testing.expectEqual(@as(usize, 3), csr.row_ptr[3]);
}

test "CSR: get element" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(f64, 1.0), csr.get(0, 0));
    try testing.expectEqual(@as(f64, 3.0), csr.get(0, 2));
    try testing.expectEqual(@as(f64, 5.0), csr.get(1, 1));
    try testing.expectEqual(@as(f64, 0.0), csr.get(0, 1)); // Zero element
    try testing.expectEqual(@as(f64, 0.0), csr.get(2, 2)); // Zero element
}

test "CSR: row iterator" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    var it = csr.rowIterator(0);
    const elem1 = it.next().?;
    try testing.expectEqual(@as(usize, 0), elem1.col);
    try testing.expectEqual(@as(f64, 1.0), elem1.value);

    const elem2 = it.next().?;
    try testing.expectEqual(@as(usize, 2), elem2.col);
    try testing.expectEqual(@as(f64, 3.0), elem2.value);

    try testing.expect(it.next() == null);
}

test "CSR: rowNnz" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(usize, 2), csr.rowNnz(0));
    try testing.expectEqual(@as(usize, 1), csr.rowNnz(1));
    try testing.expectEqual(@as(usize, 0), csr.rowNnz(2));
}

test "CSC: fromCOO basic" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    try testing.expectEqual(@as(usize, 3), csc.rows);
    try testing.expectEqual(@as(usize, 3), csc.cols);
    try testing.expectEqual(@as(usize, 3), csc.nnz());

    // Check column pointers
    try testing.expectEqual(@as(usize, 0), csc.col_ptr[0]);
    try testing.expectEqual(@as(usize, 1), csc.col_ptr[1]);
    try testing.expectEqual(@as(usize, 2), csc.col_ptr[2]);
    try testing.expectEqual(@as(usize, 3), csc.col_ptr[3]);
}

test "CSC: get element" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(2, 0, 7.0);
    try coo.append(1, 1, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    try testing.expectEqual(@as(f64, 1.0), csc.get(0, 0));
    try testing.expectEqual(@as(f64, 7.0), csc.get(2, 0));
    try testing.expectEqual(@as(f64, 5.0), csc.get(1, 1));
    try testing.expectEqual(@as(f64, 0.0), csc.get(0, 1)); // Zero element
}

test "CSC: colNnz" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(2, 0, 7.0);
    try coo.append(1, 1, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    try testing.expectEqual(@as(usize, 2), csc.colNnz(0));
    try testing.expectEqual(@as(usize, 1), csc.colNnz(1));
    try testing.expectEqual(@as(usize, 0), csc.colNnz(2));
}

test "Sparse: empty matrix" {
    var coo = COO(f64).init(testing.allocator, 5, 5);
    defer coo.deinit();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(usize, 0), csr.nnz());
    try testing.expectEqual(@as(f64, 0.0), csr.get(0, 0));
}

test "Sparse: integer types" {
    var coo = COO(i32).init(testing.allocator, 2, 2);
    defer coo.deinit();

    try coo.append(0, 0, 10);
    try coo.append(1, 1, 20);
    try coo.sort();

    var csr = try CSR(i32).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(i32, 10), csr.get(0, 0));
    try testing.expectEqual(@as(i32, 20), csr.get(1, 1));
}

test "COO: memory safety with multiple iterations" {
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var coo = COO(f64).init(testing.allocator, 100, 100);
        defer coo.deinit();

        var j: usize = 0;
        while (j < 50) : (j += 1) {
            try coo.append(j, j, @as(f64, @floatFromInt(j)));
        }

        try coo.sort();
        try testing.expectEqual(@as(usize, 50), coo.nnz());
    }
}

// ============================================================================
// Sparse Matrix Operations: Transpose and SpMV
// ============================================================================

test "CSR transpose: general 3x3 matrix" {
    // Create 3x3 matrix:
    // [1 0 3]
    // [0 5 0]
    // [7 0 9]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.append(2, 0, 7.0);
    try coo.append(2, 2, 9.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // Manually build transpose: [A^T]_ij = A_ji
    // A^T will be:
    // [1 0 7]
    // [0 5 0]
    // [3 0 9]
    var coo_t = COO(f64).init(testing.allocator, 3, 3);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 1.0); // A[0,0] -> A^T[0,0]
    try coo_t.append(0, 2, 7.0); // A[2,0] -> A^T[0,2]
    try coo_t.append(1, 1, 5.0); // A[1,1] -> A^T[1,1]
    try coo_t.append(2, 0, 3.0); // A[0,2] -> A^T[2,0]
    try coo_t.append(2, 2, 9.0); // A[2,2] -> A^T[2,2]
    try coo_t.sort();

    var csr_t = try CSR(f64).fromCOO(testing.allocator, &coo_t);
    defer csr_t.deinit();

    try testing.expectEqual(@as(usize, 3), csr_t.rows);
    try testing.expectEqual(@as(usize, 3), csr_t.cols);
    try testing.expectEqual(@as(usize, 5), csr_t.nnz());

    // Verify transposed values
    try testing.expectEqual(@as(f64, 1.0), csr_t.get(0, 0));
    try testing.expectEqual(@as(f64, 7.0), csr_t.get(0, 2));
    try testing.expectEqual(@as(f64, 5.0), csr_t.get(1, 1));
    try testing.expectEqual(@as(f64, 3.0), csr_t.get(2, 0));
    try testing.expectEqual(@as(f64, 9.0), csr_t.get(2, 2));
}

test "CSR transpose: 2x4 rectangular matrix" {
    // Create 2x4 matrix:
    // [1 2 0 3]
    // [0 4 5 0]
    var coo = COO(f64).init(testing.allocator, 2, 4);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 1, 2.0);
    try coo.append(0, 3, 3.0);
    try coo.append(1, 1, 4.0);
    try coo.append(1, 2, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // Build CSC representation of transpose (should be 4x2)
    var coo_t = COO(f64).init(testing.allocator, 4, 2);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 1.0); // (0,0) from A
    try coo_t.append(1, 0, 2.0); // (0,1) from A
    try coo_t.append(1, 1, 4.0); // (1,1) from A
    try coo_t.append(2, 1, 5.0); // (1,2) from A
    try coo_t.append(3, 0, 3.0); // (0,3) from A

    var csc_t = try CSC(f64).fromCOO(testing.allocator, &coo_t);
    defer csc_t.deinit();

    try testing.expectEqual(@as(usize, 4), csc_t.rows);
    try testing.expectEqual(@as(usize, 2), csc_t.cols);
    try testing.expectEqual(@as(usize, 5), csc_t.nnz());
}

test "CSR transpose: single element matrix" {
    // Create 1x1 matrix with single non-zero
    var coo = COO(f64).init(testing.allocator, 1, 1);
    defer coo.deinit();

    try coo.append(0, 0, 42.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    var coo_t = COO(f64).init(testing.allocator, 1, 1);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 42.0);

    var csr_t = try CSR(f64).fromCOO(testing.allocator, &coo_t);
    defer csr_t.deinit();

    try testing.expectEqual(@as(f64, 42.0), csr_t.get(0, 0));
    try testing.expectEqual(@as(usize, 1), csr_t.nnz());
}

test "CSR transpose: empty matrix" {
    // Create empty 3x3 matrix
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    var coo_t = COO(f64).init(testing.allocator, 3, 3);
    defer coo_t.deinit();

    var csr_t = try CSR(f64).fromCOO(testing.allocator, &coo_t);
    defer csr_t.deinit();

    try testing.expectEqual(@as(usize, 0), csr_t.nnz());
    try testing.expectEqual(@as(f64, 0.0), csr_t.get(0, 0));
}

test "CSC transpose: 3x3 matrix to CSR" {
    // Create 3x3 CSC matrix:
    // [1 0 3]
    // [0 5 0]
    // [7 0 9]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.append(2, 0, 7.0);
    try coo.append(2, 2, 9.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // Transpose of CSC: build COO from transposed values
    var coo_t = COO(f64).init(testing.allocator, 3, 3);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 1.0);
    try coo_t.append(1, 0, 5.0);
    try coo_t.append(2, 0, 3.0);
    try coo_t.append(0, 2, 7.0);
    try coo_t.append(2, 2, 9.0);
    try coo_t.sort();

    var csr_t = try CSR(f64).fromCOO(testing.allocator, &coo_t);
    defer csr_t.deinit();

    try testing.expectEqual(@as(usize, 3), csr_t.rows);
    try testing.expectEqual(@as(usize, 3), csr_t.cols);
    try testing.expectEqual(@as(usize, 5), csr_t.nnz());
}

test "CSC transpose: symmetric matrix" {
    // Create 3x3 symmetric matrix:
    // [2 1 0]
    // [1 3 4]
    // [0 4 5]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 2.0);
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 4.0);
    try coo.append(2, 1, 4.0);
    try coo.append(2, 2, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // Transpose should equal original (symmetric)
    var coo_t = COO(f64).init(testing.allocator, 3, 3);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 2.0);
    try coo_t.append(0, 1, 1.0);
    try coo_t.append(1, 0, 1.0);
    try coo_t.append(1, 1, 3.0);
    try coo_t.append(1, 2, 4.0);
    try coo_t.append(2, 1, 4.0);
    try coo_t.append(2, 2, 5.0);

    var csc_t = try CSC(f64).fromCOO(testing.allocator, &coo_t);
    defer csc_t.deinit();

    try testing.expectEqual(csc.nnz(), csc_t.nnz());
    // Verify symmetric property: A[i,j] = A[j,i]
    try testing.expectEqual(@as(f64, 1.0), csc_t.get(0, 1));
    try testing.expectEqual(@as(f64, 1.0), csc_t.get(1, 0));
}

// ============================================================================
// Sparse Matrix-Vector Multiplication (SpMV)
// ============================================================================

test "CSR matvec: identity matrix" {
    // Create 3x3 identity matrix
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // Create vector x = [1, 2, 3]
    const x = [_]f64{ 1.0, 2.0, 3.0 };

    // y = I * x = x = [1, 2, 3]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    // Perform y = A*x (identity case)
    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 1.0), y[0]);
    try testing.expectEqual(@as(f64, 2.0), y[1]);
    try testing.expectEqual(@as(f64, 3.0), y[2]);
}

test "CSR matvec: diagonal matrix" {
    // Create 3x3 diagonal matrix [2, 3, 5]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 2, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // x = [1, 2, 3]
    const x = [_]f64{ 1.0, 2.0, 3.0 };

    // y = D*x = [2*1, 3*2, 5*3] = [2, 6, 15]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 2.0), y[0]);
    try testing.expectEqual(@as(f64, 6.0), y[1]);
    try testing.expectEqual(@as(f64, 15.0), y[2]);
}

test "CSR matvec: general sparse matrix" {
    // Create 3x4 matrix:
    // [1 0 2 0]
    // [0 3 0 4]
    // [5 0 0 6]
    var coo = COO(f64).init(testing.allocator, 3, 4);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 3, 4.0);
    try coo.append(2, 0, 5.0);
    try coo.append(2, 3, 6.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // x = [1, 2, 3, 4]
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    // Manual calculation:
    // y[0] = 1*1 + 2*3 = 1 + 6 = 7
    // y[1] = 3*2 + 4*4 = 6 + 16 = 22
    // y[2] = 5*1 + 6*4 = 5 + 24 = 29
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 7.0), y[0]);
    try testing.expectEqual(@as(f64, 22.0), y[1]);
    try testing.expectEqual(@as(f64, 29.0), y[2]);
}

test "CSR matvec: single row matrix" {
    // Create 1x4 matrix: [2 0 3 0]
    var coo = COO(f64).init(testing.allocator, 1, 4);
    defer coo.deinit();

    try coo.append(0, 0, 2.0);
    try coo.append(0, 2, 3.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // x = [1, 2, 3, 4]
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    // y = [2*1 + 3*3] = [11]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 1);
    defer allocator.free(y);

    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 11.0), y[0]);
}

test "CSR matvec: single column matrix" {
    // Create 3x1 matrix: [5]
    //                    [2]
    //                    [7]
    var coo = COO(f64).init(testing.allocator, 3, 1);
    defer coo.deinit();

    try coo.append(0, 0, 5.0);
    try coo.append(1, 0, 2.0);
    try coo.append(2, 0, 7.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // x = [3]
    const x = [_]f64{3.0};

    // y = [5*3, 2*3, 7*3] = [15, 6, 21]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 15.0), y[0]);
    try testing.expectEqual(@as(f64, 6.0), y[1]);
    try testing.expectEqual(@as(f64, 21.0), y[2]);
}

test "CSR matvec: empty matrix" {
    // Create empty 3x3 matrix
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const x = [_]f64{ 1.0, 2.0, 3.0 };

    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    // y = A*x where A is zero matrix should be zero vector
    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 0.0), y[0]);
    try testing.expectEqual(@as(f64, 0.0), y[1]);
    try testing.expectEqual(@as(f64, 0.0), y[2]);
}

test "CSC matvec: identity matrix" {
    // Create 3x3 identity matrix via CSC
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // x = [1, 2, 3]
    const x = [_]f64{ 1.0, 2.0, 3.0 };

    // y = I * x = x = [1, 2, 3]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    // SpMV with CSC: column-wise accumulation
    for (0..csc.rows) |i| {
        y[i] = 0.0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(f64, 1.0), y[0]);
    try testing.expectEqual(@as(f64, 2.0), y[1]);
    try testing.expectEqual(@as(f64, 3.0), y[2]);
}

test "CSC matvec: diagonal matrix" {
    // Create 3x3 diagonal matrix [2, 3, 5]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 2, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // x = [1, 2, 3]
    const x = [_]f64{ 1.0, 2.0, 3.0 };

    // y = D*x = [2*1, 3*2, 5*3] = [2, 6, 15]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csc.rows) |i| {
        y[i] = 0.0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(f64, 2.0), y[0]);
    try testing.expectEqual(@as(f64, 6.0), y[1]);
    try testing.expectEqual(@as(f64, 15.0), y[2]);
}

test "CSC matvec: general sparse matrix" {
    // Create 3x4 matrix:
    // [1 0 2 0]
    // [0 3 0 4]
    // [5 0 0 6]
    var coo = COO(f64).init(testing.allocator, 3, 4);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 3, 4.0);
    try coo.append(2, 0, 5.0);
    try coo.append(2, 3, 6.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // x = [1, 2, 3, 4]
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    // SpMV: y = A*x
    // y[0] = A[0,0]*x[0] + A[0,1]*x[1] + A[0,2]*x[2] + A[0,3]*x[3] = 1*1 + 0*2 + 2*3 + 0*4 = 1 + 6 = 7
    // y[1] = A[1,0]*x[0] + A[1,1]*x[1] + A[1,2]*x[2] + A[1,3]*x[3] = 0*1 + 3*2 + 0*3 + 4*4 = 6 + 16 = 22
    // y[2] = A[2,0]*x[0] + A[2,1]*x[1] + A[2,2]*x[2] + A[2,3]*x[3] = 5*1 + 0*2 + 0*3 + 6*4 = 5 + 24 = 29
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csc.rows) |i| {
        y[i] = 0.0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(f64, 7.0), y[0]);
    try testing.expectEqual(@as(f64, 22.0), y[1]);
    try testing.expectEqual(@as(f64, 29.0), y[2]);
}

test "CSC matvec: empty matrix" {
    // Create empty 3x3 matrix
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const x = [_]f64{ 1.0, 2.0, 3.0 };

    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csc.rows) |i| {
        y[i] = 0.0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(f64, 0.0), y[0]);
    try testing.expectEqual(@as(f64, 0.0), y[1]);
    try testing.expectEqual(@as(f64, 0.0), y[2]);
}

test "CSC matvec: integer types" {
    // Create 2x3 matrix with i32:
    // [1 2 0]
    // [0 3 4]
    var coo = COO(i32).init(testing.allocator, 2, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1);
    try coo.append(0, 1, 2);
    try coo.append(1, 1, 3);
    try coo.append(1, 2, 4);

    var csc = try CSC(i32).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // x = [5, 6, 7]
    const x = [_]i32{ 5, 6, 7 };

    // y[0] = 1*5 + 2*6 = 5 + 12 = 17
    // y[1] = 3*6 + 4*7 = 18 + 28 = 46
    const allocator = testing.allocator;
    const y = try allocator.alloc(i32, 2);
    defer allocator.free(y);

    for (0..csc.rows) |i| {
        y[i] = 0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(i32, 17), y[0]);
    try testing.expectEqual(@as(i32, 46), y[1]);
}

test "COO: validate valid matrix" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 2, 2.0);
    try coo.append(2, 1, 3.0);

    // Should pass validation
    try coo.validate();
}

test "COO: validate detects out-of-bounds row" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(3, 0, 2.0); // row 3 is out of bounds for 3x3 matrix

    // Should fail validation
    const result = coo.validate();
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "COO: validate detects out-of-bounds column" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 3, 2.0); // col 3 is out of bounds for 3x3 matrix

    // Should fail validation
    const result = coo.validate();
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "CSR: validate valid matrix" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // Should pass validation
    try csr.validate();
}

test "CSC: validate valid matrix" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);
    try coo.sort();

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // Should pass validation
    try csc.validate();
}
