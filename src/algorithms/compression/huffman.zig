const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const PriorityQueue = std.PriorityQueue;

/// Huffman Coding - Optimal prefix-free encoding for data compression
///
/// Algorithm: Builds optimal prefix codes based on symbol frequencies
/// Properties:
///   - Prefix-free: No codeword is a prefix of another
///   - Optimal: Minimizes average code length for given frequencies
///   - Variable length: Frequent symbols get shorter codes
///
/// Time complexity:
///   - encode(): O(n log k) where n = data length, k = alphabet size
///   - decode(): O(m) where m = encoded bits
///   - buildTree(): O(k log k) for tree construction
///
/// Space complexity: O(k) for tree and codebook
///
/// Use cases:
///   - ZIP/GZIP compression (DEFLATE = LZ77 + Huffman)
///   - JPEG/PNG image compression
///   - MP3/AAC audio compression
///   - Text file compression
///   - Prefix-free code generation
///
/// Trade-offs:
///   - vs Arithmetic: Simpler, faster, but uses whole bits (less optimal for skewed distributions)
///   - vs RLE: Better for arbitrary data (RLE only works for runs)
///   - vs LZ77: Entropy coding (no dictionary), complements LZ77 in DEFLATE
///
/// Reference:
///   - Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes"
///   - Used in ZIP, GZIP, PNG, JPEG, MP3

/// Huffman tree node
const Node = struct {
    freq: usize,
    byte: ?u8, // null for internal nodes
    left: ?*Node,
    right: ?Node,

    fn compare(_: void, a: *Node, b: *Node) std.math.Order {
        return std.math.order(a.freq, b.freq);
    }
};

/// Huffman encoder/decoder
pub const HuffmanCoding = struct {
    allocator: Allocator,
    root: ?*Node,
    codebook: AutoHashMap(u8, Code),

    const Code = struct {
        bits: u32, // Bit pattern
        length: u8, // Number of valid bits
    };

    /// Initialize empty Huffman coding
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn init(allocator: Allocator) HuffmanCoding {
        return .{
            .allocator = allocator,
            .root = null,
            .codebook = AutoHashMap(u8, Code).init(allocator),
        };
    }

    /// Free all resources
    ///
    /// Time: O(k) where k = alphabet size
    /// Space: O(1)
    pub fn deinit(self: *HuffmanCoding) void {
        if (self.root) |root| {
            self.freeNode(root);
        }
        self.codebook.deinit();
    }

    fn freeNode(self: *HuffmanCoding, node: *Node) void {
        if (node.left) |left| {
            self.freeNode(left);
        }
        if (node.right) |right| {
            self.freeNode(right);
        }
        self.allocator.destroy(node);
    }

    /// Build Huffman tree from frequency table
    ///
    /// Time: O(k log k) where k = alphabet size
    /// Space: O(k)
    pub fn buildFromFrequencies(self: *HuffmanCoding, frequencies: AutoHashMap(u8, usize)) !void {
        // Clear existing tree
        if (self.root) |root| {
            self.freeNode(root);
        }
        self.codebook.clearRetainingCapacity();

        // Edge case: empty
        if (frequencies.count() == 0) {
            self.root = null;
            return;
        }

        // Edge case: single symbol
        if (frequencies.count() == 1) {
            var iter = frequencies.iterator();
            const entry = iter.next().?;
            const node = try self.allocator.create(Node);
            node.* = .{
                .freq = entry.value_ptr.*,
                .byte = entry.key_ptr.*,
                .left = null,
                .right = null,
            };
            self.root = node;
            // Single symbol gets code "0"
            try self.codebook.put(entry.key_ptr.*, .{ .bits = 0, .length = 1 });
            return;
        }

        // Priority queue of nodes (min-heap by frequency)
        var pq = PriorityQueue(*Node, void, Node.compare).init(self.allocator, {});
        defer pq.deinit();

        // Create leaf nodes
        var iter = frequencies.iterator();
        while (iter.next()) |entry| {
            const node = try self.allocator.create(Node);
            node.* = .{
                .freq = entry.value_ptr.*,
                .byte = entry.key_ptr.*,
                .left = null,
                .right = null,
            };
            try pq.add(node);
        }

        // Build tree bottom-up
        while (pq.count() > 1) {
            const left = pq.remove();
            const right = pq.remove();

            const parent = try self.allocator.create(Node);
            parent.* = .{
                .freq = left.freq + right.freq,
                .byte = null,
                .left = left,
                .right = right,
            };
            try pq.add(parent);
        }

        self.root = pq.remove();

        // Generate codebook
        try self.generateCodes(self.root.?, 0, 0);
    }

    fn generateCodes(self: *HuffmanCoding, node: *Node, bits: u32, length: u8) !void {
        if (node.byte) |byte| {
            // Leaf node
            try self.codebook.put(byte, .{ .bits = bits, .length = length });
        } else {
            // Internal node
            if (node.left) |left| {
                try self.generateCodes(left, bits << 1, length + 1);
            }
            if (node.right) |right| {
                try self.generateCodes(right, (bits << 1) | 1, length + 1);
            }
        }
    }

    /// Build Huffman tree from data
    ///
    /// Time: O(n + k log k) where n = data length, k = alphabet size
    /// Space: O(k)
    pub fn buildFromData(self: *HuffmanCoding, data: []const u8) !void {
        var frequencies = AutoHashMap(u8, usize).init(self.allocator);
        defer frequencies.deinit();

        for (data) |byte| {
            const entry = try frequencies.getOrPut(byte);
            if (!entry.found_existing) {
                entry.value_ptr.* = 0;
            }
            entry.value_ptr.* += 1;
        }

        try self.buildFromFrequencies(frequencies);
    }

    /// Encode data using Huffman codes
    ///
    /// Returns encoded bits as bytes (bit-packed)
    /// Format: [encoded_bits_length: u32][encoded_bits: bytes]
    ///
    /// Time: O(n) where n = data length
    /// Space: O(n) for encoded output
    pub fn encode(self: *HuffmanCoding, data: []const u8) ![]u8 {
        if (data.len == 0) {
            const result = try self.allocator.alloc(u8, 4);
            std.mem.writeInt(u32, result[0..4], 0, .little);
            return result;
        }

        // Calculate total bits needed
        var total_bits: usize = 0;
        for (data) |byte| {
            const code = self.codebook.get(byte) orelse return error.UnknownSymbol;
            total_bits += code.length;
        }

        // Allocate output (4 bytes for length + bit-packed data)
        const byte_count = (total_bits + 7) / 8;
        const result = try self.allocator.alloc(u8, 4 + byte_count);
        @memset(result, 0);

        // Write bit count
        std.mem.writeInt(u32, result[0..4], @intCast(total_bits), .little);

        // Encode bits
        var bit_pos: usize = 0;
        for (data) |byte| {
            const code = self.codebook.get(byte).?;
            const bits = code.bits;
            var len = code.length;

            while (len > 0) {
                len -= 1;
                const bit = @as(u8, @intCast((bits >> @intCast(len)) & 1));
                const byte_idx = bit_pos / 8;
                const bit_idx = @as(u3, @intCast(7 - (bit_pos % 8)));
                result[4 + byte_idx] |= bit << bit_idx;
                bit_pos += 1;
            }
        }

        return result;
    }

    /// Decode Huffman-encoded data
    ///
    /// Time: O(m) where m = encoded bits
    /// Space: O(n) for decoded output
    pub fn decode(self: *HuffmanCoding, encoded: []const u8) ![]u8 {
        if (encoded.len < 4) return error.InvalidEncodedData;

        const total_bits = std.mem.readInt(u32, encoded[0..4], .little);
        if (total_bits == 0) {
            return try self.allocator.alloc(u8, 0);
        }

        if (self.root == null) return error.NoTreeBuilt;

        var result = ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        var node = self.root.?;
        var bit_pos: usize = 0;

        while (bit_pos < total_bits) {
            // Single symbol case
            if (node.left == null and node.right == null) {
                if (node.byte) |byte| {
                    try result.append(byte);
                    bit_pos += 1;
                }
                continue;
            }

            // Read bit
            const byte_idx = bit_pos / 8;
            const bit_idx = @as(u3, @intCast(7 - (bit_pos % 8)));
            const bit = (encoded[4 + byte_idx] >> bit_idx) & 1;

            // Traverse tree
            if (bit == 0) {
                node = node.left orelse return error.InvalidEncodedData;
            } else {
                node = node.right orelse return error.InvalidEncodedData;
            }

            // Reached leaf
            if (node.byte) |byte| {
                try result.append(byte);
                node = self.root.?;
            }

            bit_pos += 1;
        }

        return result.toOwnedSlice();
    }

    /// Get code for a symbol
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn getCode(self: *HuffmanCoding, byte: u8) ?Code {
        return self.codebook.get(byte);
    }

    /// Get number of symbols in codebook
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn symbolCount(self: *HuffmanCoding) usize {
        return self.codebook.count();
    }

    /// Calculate average code length for given data
    ///
    /// Time: O(n) where n = data length
    /// Space: O(1)
    pub fn averageCodeLength(self: *HuffmanCoding, data: []const u8) !f64 {
        if (data.len == 0) return 0.0;

        var total_bits: usize = 0;
        for (data) |byte| {
            const code = self.codebook.get(byte) orelse return error.UnknownSymbol;
            total_bits += code.length;
        }

        return @as(f64, @floatFromInt(total_bits)) / @as(f64, @floatFromInt(data.len));
    }

    /// Calculate compression ratio (original / compressed)
    ///
    /// Time: O(n)
    /// Space: O(1)
    pub fn compressionRatio(self: *HuffmanCoding, data: []const u8) !f64 {
        if (data.len == 0) return 1.0;

        const avg_len = try self.averageCodeLength(data);
        return 8.0 / avg_len; // 8 bits per byte original
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Huffman: empty data" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    const data = "";
    try huff.buildFromData(data);

    const encoded = try huff.encode(data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, data, decoded);
}

test "Huffman: single symbol" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    const data = "aaaa";
    try huff.buildFromData(data);

    try std.testing.expectEqual(@as(usize, 1), huff.symbolCount());

    const encoded = try huff.encode(data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, data, decoded);
}

test "Huffman: two symbols equal frequency" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    const data = "aabb";
    try huff.buildFromData(data);

    try std.testing.expectEqual(@as(usize, 2), huff.symbolCount());

    // Both symbols should have 1-bit codes
    const code_a = huff.getCode('a').?;
    const code_b = huff.getCode('b').?;
    try std.testing.expectEqual(@as(u8, 1), code_a.length);
    try std.testing.expectEqual(@as(u8, 1), code_b.length);

    const encoded = try huff.encode(data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, data, decoded);
}

test "Huffman: skewed distribution" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    const data = "aaaaaabcd"; // a:6, b:1, c:1, d:1
    try huff.buildFromData(data);

    try std.testing.expectEqual(@as(usize, 4), huff.symbolCount());

    // 'a' should have shortest code (most frequent)
    const code_a = huff.getCode('a').?;
    const code_b = huff.getCode('b').?;
    try std.testing.expect(code_a.length <= code_b.length);

    const encoded = try huff.encode(data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, data, decoded);
}

test "Huffman: roundtrip ASCII" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    const data = "hello world";
    try huff.buildFromData(data);

    const encoded = try huff.encode(data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, data, decoded);
}

test "Huffman: roundtrip binary data" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    const data = [_]u8{ 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 0 };
    try huff.buildFromData(&data);

    const encoded = try huff.encode(&data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, &data, decoded);
}

test "Huffman: compression ratio" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    // Highly skewed data should compress well
    const data = "aaaaaaaabbcd";
    try huff.buildFromData(data);

    const ratio = try huff.compressionRatio(data);
    try std.testing.expect(ratio > 1.0); // Should compress
}

test "Huffman: average code length" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    const data = "aabb";
    try huff.buildFromData(data);

    const avg = try huff.averageCodeLength(data);
    try std.testing.expectApproxEqAbs(1.0, avg, 0.01); // Should be 1 bit/symbol
}

test "Huffman: long text" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    // Realistic text with varying frequencies
    const data = "the quick brown fox jumps over the lazy dog";
    try huff.buildFromData(data);

    const encoded = try huff.encode(data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, data, decoded);

    // Should achieve some compression
    const ratio = try huff.compressionRatio(data);
    try std.testing.expect(ratio > 1.0);
}

test "Huffman: all 256 bytes" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    var data: [256]u8 = undefined;
    for (&data, 0..) |*byte, i| {
        byte.* = @intCast(i);
    }

    try huff.buildFromData(&data);

    const encoded = try huff.encode(&data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, &data, decoded);
}

test "Huffman: unknown symbol error" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    try huff.buildFromData("abc");

    // Try to encode symbol not in codebook
    const result = huff.encode("abcd");
    try std.testing.expectError(error.UnknownSymbol, result);
}

test "Huffman: rebuild tree" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    // First tree
    try huff.buildFromData("aabb");
    try std.testing.expectEqual(@as(usize, 2), huff.symbolCount());

    // Rebuild with different data
    try huff.buildFromData("xyz");
    try std.testing.expectEqual(@as(usize, 3), huff.symbolCount());

    // Old symbols should not be in new codebook
    try std.testing.expect(huff.getCode('a') == null);
    try std.testing.expect(huff.getCode('x') != null);
}

test "Huffman: memory safety" {
    const allocator = std.testing.allocator;

    for (0..10) |_| {
        var huff = HuffmanCoding.init(allocator);
        defer huff.deinit();

        const data = "compression test data with various symbols";
        try huff.buildFromData(data);

        const encoded = try huff.encode(data);
        defer allocator.free(encoded);

        const decoded = try huff.decode(encoded);
        defer allocator.free(decoded);
    }
}

test "Huffman: prefix-free property" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    const data = "abcdef";
    try huff.buildFromData(data);

    // No code should be a prefix of another
    // This is guaranteed by Huffman construction, but verify via roundtrip
    const encoded = try huff.encode(data);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, data, decoded);
}

test "Huffman: large data" {
    const allocator = std.testing.allocator;

    var huff = HuffmanCoding.init(allocator);
    defer huff.deinit();

    // Generate large data with realistic distribution
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..1000) |_| {
        // Skewed distribution: 'a' very common, others rare
        const r = random.int(u8) % 100;
        const byte = if (r < 70) 'a' else if (r < 85) 'b' else if (r < 95) 'c' else 'd';
        try data.append(byte);
    }

    try huff.buildFromData(data.items);

    const encoded = try huff.encode(data.items);
    defer allocator.free(encoded);

    const decoded = try huff.decode(encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, data.items, decoded);

    // Should compress well due to skewed distribution
    const ratio = try huff.compressionRatio(data.items);
    try std.testing.expect(ratio > 1.5);
}
