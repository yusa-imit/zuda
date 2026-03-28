const std = @import("std");
const testing = std.testing;

/// Huffman tree node
pub const HuffmanNode = struct {
    symbol: ?u8, // null for internal nodes
    frequency: u64,
    left: ?*HuffmanNode,
    right: ?*HuffmanNode,

    pub fn isLeaf(self: *const HuffmanNode) bool {
        return self.left == null and self.right == null;
    }
};

/// Huffman code mapping (symbol -> bit string)
pub const HuffmanCode = struct {
    symbol: u8,
    code: []const u8, // '0' and '1' characters
    length: usize,
};

/// Builds Huffman coding tree from symbol frequencies
///
/// Algorithm: Min-heap of nodes, repeatedly merge two lowest-frequency nodes
///
/// Time: O(n log n) — n heap operations
/// Space: O(n) — tree storage
///
/// Example:
/// ```zig
/// const symbols = "aabbbcccc";
/// var tree = try buildHuffmanTree(allocator, symbols);
/// defer destroyHuffmanTree(allocator, tree);
/// ```
pub fn buildHuffmanTree(
    allocator: std.mem.Allocator,
    data: []const u8,
) !*HuffmanNode {
    if (data.len == 0) return error.EmptyData;

    // Count frequencies
    var freq_map = std.AutoHashMap(u8, u64).init(allocator);
    defer freq_map.deinit();

    for (data) |byte| {
        const count = freq_map.get(byte) orelse 0;
        try freq_map.put(byte, count + 1);
    }

    if (freq_map.count() == 1) {
        // Special case: single unique symbol
        var it = freq_map.iterator();
        const entry = it.next().?;
        const node = try allocator.create(HuffmanNode);
        node.* = .{
            .symbol = entry.key_ptr.*,
            .frequency = entry.value_ptr.*,
            .left = null,
            .right = null,
        };
        return node;
    }

    // Create min-heap (priority queue) of nodes
    const Context = struct {
        pub fn lessThan(_: @This(), a: *HuffmanNode, b: *HuffmanNode) std.math.Order {
            return std.math.order(a.frequency, b.frequency);
        }
    };

    var heap = std.PriorityQueue(*HuffmanNode, Context, Context.lessThan).init(allocator, .{});
    defer heap.deinit();

    // Initialize heap with leaf nodes
    var it = freq_map.iterator();
    while (it.next()) |entry| {
        const node = try allocator.create(HuffmanNode);
        node.* = .{
            .symbol = entry.key_ptr.*,
            .frequency = entry.value_ptr.*,
            .left = null,
            .right = null,
        };
        try heap.add(node);
    }

    // Build tree by merging nodes
    while (heap.count() > 1) {
        const left = heap.remove();
        const right = heap.remove();

        const parent = try allocator.create(HuffmanNode);
        parent.* = .{
            .symbol = null,
            .frequency = left.frequency + right.frequency,
            .left = left,
            .right = right,
        };
        try heap.add(parent);
    }

    return heap.remove();
}

/// Generates Huffman codes for all symbols in the tree
///
/// Time: O(n) — tree traversal
/// Space: O(n) — code storage
pub fn generateHuffmanCodes(
    allocator: std.mem.Allocator,
    root: *HuffmanNode,
) !std.ArrayList(HuffmanCode) {
    var codes = std.ArrayList(HuffmanCode).init(allocator);
    errdefer {
        for (codes.items) |code| {
            allocator.free(code.code);
        }
        codes.deinit();
    }

    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    try generateCodesRecursive(allocator, root, &buffer, &codes);
    return codes;
}

fn generateCodesRecursive(
    allocator: std.mem.Allocator,
    node: *HuffmanNode,
    current_code: *std.ArrayList(u8),
    codes: *std.ArrayList(HuffmanCode),
) !void {
    if (node.isLeaf()) {
        if (node.symbol) |sym| {
            // Special case: single symbol tree
            const code_str = if (current_code.items.len == 0)
                try allocator.dupe(u8, "0")
            else
                try allocator.dupe(u8, current_code.items);

            try codes.append(.{
                .symbol = sym,
                .code = code_str,
                .length = code_str.len,
            });
        }
        return;
    }

    // Traverse left (append '0')
    if (node.left) |left| {
        try current_code.append('0');
        try generateCodesRecursive(allocator, left, current_code, codes);
        _ = current_code.pop();
    }

    // Traverse right (append '1')
    if (node.right) |right| {
        try current_code.append('1');
        try generateCodesRecursive(allocator, right, current_code, codes);
        _ = current_code.pop();
    }
}

/// Encodes data using Huffman codes
///
/// Time: O(n) — single pass over data
/// Space: O(n) — encoded output
pub fn huffmanEncode(
    allocator: std.mem.Allocator,
    data: []const u8,
    codes: []const HuffmanCode,
) ![]const u8 {
    // Build symbol->code lookup
    var code_map = std.AutoHashMap(u8, []const u8).init(allocator);
    defer code_map.deinit();

    for (codes) |code| {
        try code_map.put(code.symbol, code.code);
    }

    // Encode data
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    for (data) |byte| {
        const code = code_map.get(byte) orelse return error.SymbolNotFound;
        try result.appendSlice(code);
    }

    return result.toOwnedSlice();
}

/// Decodes Huffman-encoded data
///
/// Time: O(n) — single pass over encoded data
/// Space: O(n) — decoded output
pub fn huffmanDecode(
    allocator: std.mem.Allocator,
    encoded: []const u8,
    root: *HuffmanNode,
) ![]const u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    if (encoded.len == 0) {
        return result.toOwnedSlice();
    }

    var current = root;
    for (encoded) |bit| {
        if (bit == '0') {
            current = current.left orelse return error.InvalidEncoding;
        } else if (bit == '1') {
            current = current.right orelse return error.InvalidEncoding;
        } else {
            return error.InvalidBitCharacter;
        }

        if (current.isLeaf()) {
            try result.append(current.symbol orelse return error.InvalidTree);
            current = root; // Reset to root for next symbol
        }
    }

    if (current != root) {
        return error.IncompleteEncoding;
    }

    return result.toOwnedSlice();
}

/// Destroys Huffman tree and frees all nodes
///
/// Time: O(n) — tree traversal
/// Space: O(log n) — recursion stack
pub fn destroyHuffmanTree(allocator: std.mem.Allocator, root: *HuffmanNode) void {
    if (root.left) |left| destroyHuffmanTree(allocator, left);
    if (root.right) |right| destroyHuffmanTree(allocator, right);
    allocator.destroy(root);
}

// Tests
test "huffman - basic encoding/decoding" {
    const data = "aabbbcccc";

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer {
        for (codes.items) |code| {
            testing.allocator.free(code.code);
        }
        codes.deinit();
    }

    // Verify codes are prefix-free
    for (codes.items, 0..) |code1, i| {
        for (codes.items[i + 1 ..]) |code2| {
            try testing.expect(!std.mem.startsWith(u8, code1.code, code2.code));
            try testing.expect(!std.mem.startsWith(u8, code2.code, code1.code));
        }
    }

    const encoded = try huffmanEncode(testing.allocator, data, codes.items);
    defer testing.allocator.free(encoded);

    const decoded = try huffmanDecode(testing.allocator, encoded, tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualStrings(data, decoded);
}

test "huffman - single character" {
    const data = "aaaa";

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer {
        for (codes.items) |code| {
            testing.allocator.free(code.code);
        }
        codes.deinit();
    }

    try testing.expectEqual(@as(usize, 1), codes.items.len);
    try testing.expectEqual(@as(u8, 'a'), codes.items[0].symbol);

    const encoded = try huffmanEncode(testing.allocator, data, codes.items);
    defer testing.allocator.free(encoded);

    const decoded = try huffmanDecode(testing.allocator, encoded, tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualStrings(data, decoded);
}

test "huffman - compression ratio" {
    const data = "aaaabbbbccccddddeeeeffffgggghhhhiiiijjjj";

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer {
        for (codes.items) |code| {
            testing.allocator.free(code.code);
        }
        codes.deinit();
    }

    const encoded = try huffmanEncode(testing.allocator, data, codes.items);
    defer testing.allocator.free(encoded);

    // Huffman encoding should be shorter than fixed-width encoding
    const fixed_bits = data.len * 4; // 4 bits for 10 symbols
    const huffman_bits = encoded.len;
    try testing.expect(huffman_bits <= fixed_bits);

    const decoded = try huffmanDecode(testing.allocator, encoded, tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualStrings(data, decoded);
}

test "huffman - empty data" {
    const data: []const u8 = "";
    try testing.expectError(error.EmptyData, buildHuffmanTree(testing.allocator, data));
}

test "huffman - all unique characters" {
    const data = "abcdef";

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer {
        for (codes.items) |code| {
            testing.allocator.free(code.code);
        }
        codes.deinit();
    }

    try testing.expectEqual(@as(usize, 6), codes.items.len);

    const encoded = try huffmanEncode(testing.allocator, data, codes.items);
    defer testing.allocator.free(encoded);

    const decoded = try huffmanDecode(testing.allocator, encoded, tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualStrings(data, decoded);
}
