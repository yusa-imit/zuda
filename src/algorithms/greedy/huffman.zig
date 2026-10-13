//! Huffman coding: tree construction from byte frequencies, code generation, encoding and
//! decoding.
//!
//! Codes are text bit strings of '0' and '1' characters, which keeps the API easy to inspect;
//! it is not a packed bit format. Symbols are bytes, so a tree has at most 256 leaves, 511 nodes
//! and depth 255; every traversal here is iterative and bounded by those limits.
//!
//! Ownership: a tree from `buildHuffmanTree` is released with `destroyHuffmanTree` (it must be
//! a tree of that origin, at most `nodes_max` nodes). Each `HuffmanCode.code` and the list that
//! holds them belong to the caller, as do the slices returned by `huffmanEncode` and
//! `huffmanDecode`; all are allocated with the caller's `gpa`. On error nothing is leaked.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

/// Number of distinct byte symbols, and therefore the maximum number of leaves.
const symbols_max: usize = 256;
/// A binary tree with `symbols_max` leaves has `2 * symbols_max - 1` nodes.
const nodes_max: usize = 2 * symbols_max - 1;

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

const NodeQueue = std.PriorityQueue(*HuffmanNode, void, nodeOrder);

fn nodeOrder(_: void, a: *HuffmanNode, b: *HuffmanNode) std.math.Order {
    return std.math.order(a.frequency, b.frequency);
}

/// Builds Huffman coding tree from symbol frequencies
///
/// Algorithm: Min-heap of nodes, repeatedly merge two lowest-frequency nodes. Leaves enter the
/// heap in symbol order, so the same data always yields the same tree.
///
/// Time: O(n log n) — n heap operations
/// Space: O(n) — tree storage
///
/// Returns `error.EmptyData` for empty input. Data with a single distinct symbol yields a lone
/// leaf.
///
/// Example:
/// ```zig
/// const symbols = "aabbbcccc";
/// var tree = try buildHuffmanTree(gpa, symbols);
/// defer destroyHuffmanTree(gpa, tree);
/// ```
pub fn buildHuffmanTree(
    gpa: Allocator,
    data: []const u8,
) !*HuffmanNode {
    if (data.len == 0) return error.EmptyData;

    // Count frequencies
    var frequencies = [_]u64{0} ** symbols_max;
    for (data) |byte| frequencies[byte] += 1;

    var heap = NodeQueue.initContext({});
    defer heap.deinit(gpa);

    // On failure every subtree still queued is destroyed. The queue never holds more than one
    // entry per symbol, so after the reservation below pushes do not allocate.
    errdefer destroyQueuedTrees(gpa, &heap);

    try heap.ensureTotalCapacityPrecise(gpa, symbols_max);

    // Initialize heap with leaf nodes
    for (frequencies, 0..) |frequency, symbol| {
        if (frequency == 0) continue;

        const node = try gpa.create(HuffmanNode);
        node.* = .{
            .symbol = @intCast(symbol),
            .frequency = frequency,
            .left = null,
            .right = null,
        };
        try pushOrDestroy(gpa, &heap, node);
    }

    // Build tree by merging nodes; each merge removes one queue entry.
    for (0..symbols_max) |_| {
        if (heap.count() <= 1) break;

        const left = heap.pop().?;
        const right = heap.pop().?;

        const parent = gpa.create(HuffmanNode) catch |err| {
            destroyHuffmanTree(gpa, left);
            destroyHuffmanTree(gpa, right);
            return err;
        };
        parent.* = .{
            .symbol = null,
            .frequency = left.frequency + right.frequency,
            .left = left,
            .right = right,
        };
        try pushOrDestroy(gpa, &heap, parent);
    }

    assert(heap.count() == 1);
    return heap.pop().?;
}

/// Pushes `node`; if the queue cannot grow, the subtree is destroyed so nothing leaks.
fn pushOrDestroy(gpa: Allocator, heap: *NodeQueue, node: *HuffmanNode) !void {
    assert(heap.count() < symbols_max);
    heap.push(gpa, node) catch |err| {
        destroyHuffmanTree(gpa, node);
        return err;
    };
}

fn destroyQueuedTrees(gpa: Allocator, heap: *NodeQueue) void {
    for (0..symbols_max) |_| {
        const node = heap.pop() orelse break;
        destroyHuffmanTree(gpa, node);
    }
    assert(heap.count() == 0);
}

/// Generates Huffman codes for all symbols in the tree
///
/// Codes come out in left-to-right leaf order ('0' = left, '1' = right). A lone-leaf tree gets
/// the code "0".
///
/// Time: O(n) — tree traversal
/// Space: O(n) — code storage
pub fn generateHuffmanCodes(
    gpa: Allocator,
    root: *HuffmanNode,
) !ArrayList(HuffmanCode) {
    var codes: ArrayList(HuffmanCode) = .empty;
    errdefer {
        for (codes.items) |code| {
            gpa.free(code.code);
        }
        codes.deinit(gpa);
    }

    // The path from the root has at most `symbols_max - 1` edges.
    var path: ArrayList(u8) = .empty;
    defer path.deinit(gpa);

    try path.ensureTotalCapacityPrecise(gpa, symbols_max);

    const Frame = struct { node: *HuffmanNode, branch: u8 };
    var frames = [_]Frame{.{ .node = root, .branch = 0 }} ** symbols_max;
    var depth: usize = 1;

    // A leaf takes one visit, an internal node three (left, right, done).
    for (0..3 * nodes_max) |_| {
        if (depth == 0) break;
        assert(path.items.len + 1 == depth);

        const frame = &frames[depth - 1];
        if (frame.node.isLeaf()) {
            try appendLeafCode(gpa, &codes, frame.node, path.items);
            depth -= 1;
            if (depth > 0) _ = path.pop();
            continue;
        }

        const branch = frame.branch;
        frame.branch += 1;
        const child = switch (branch) {
            0 => frame.node.left,
            1 => frame.node.right,
            else => {
                depth -= 1;
                if (depth > 0) _ = path.pop();
                continue;
            },
        };
        if (child) |next| {
            path.appendAssumeCapacity(if (branch == 0) '0' else '1');
            frames[depth] = .{ .node = next, .branch = 0 };
            depth += 1;
        }
    }

    assert(depth == 0);
    return codes;
}

fn appendLeafCode(
    gpa: Allocator,
    codes: *ArrayList(HuffmanCode),
    leaf: *const HuffmanNode,
    path: []const u8,
) !void {
    assert(leaf.isLeaf());
    assert(path.len < symbols_max);

    // Internal nodes without a symbol carry no code.
    const symbol = leaf.symbol orelse return;

    // Special case: single symbol tree
    const code_str = if (path.len == 0) try gpa.dupe(u8, "0") else try gpa.dupe(u8, path);
    errdefer gpa.free(code_str);

    try codes.append(gpa, .{
        .symbol = symbol,
        .code = code_str,
        .length = code_str.len,
    });
}

/// Encodes data using Huffman codes
///
/// Returns `error.SymbolNotFound` if a byte of `data` has no code. If `codes` lists a symbol
/// more than once, the last entry wins.
///
/// Time: O(n) — single pass over data
/// Space: O(n) — encoded output
pub fn huffmanEncode(
    gpa: Allocator,
    data: []const u8,
    codes: []const HuffmanCode,
) ![]const u8 {
    // Build symbol->code lookup
    var code_table = [_]?[]const u8{null} ** symbols_max;
    for (codes) |code| code_table[code.symbol] = code.code;

    // Size the output exactly, so the encoding pass cannot fail.
    var encoded_len: usize = 0;
    for (data) |byte| {
        const code = code_table[byte] orelse return error.SymbolNotFound;
        encoded_len += code.len;
    }

    const result = try gpa.alloc(u8, encoded_len);
    var written: usize = 0;
    for (data) |byte| {
        const code = code_table[byte].?;
        @memcpy(result[written..][0..code.len], code);
        written += code.len;
    }

    assert(written == encoded_len);
    return result;
}

/// Decodes Huffman-encoded data
///
/// Errors: `InvalidBitCharacter` for anything but '0' and '1', `InvalidEncoding` for a path that
/// leaves the tree, `InvalidTree` for a leaf without symbol, `IncompleteEncoding` when the
/// input ends in the middle of a code. With a lone-leaf tree every '0' decodes one symbol.
///
/// Time: O(n) — single pass over encoded data
/// Space: O(n) — decoded output
pub fn huffmanDecode(
    gpa: Allocator,
    encoded: []const u8,
    root: *HuffmanNode,
) ![]const u8 {
    var result: ArrayList(u8) = .empty;
    errdefer result.deinit(gpa);

    if (encoded.len == 0) {
        return result.toOwnedSlice(gpa);
    }

    // Every decoded symbol consumes at least one bit.
    try result.ensureTotalCapacityPrecise(gpa, encoded.len);

    var current = root;
    for (encoded) |bit| {
        if (root.isLeaf()) {
            // Lone-leaf tree: the code of its only symbol is "0".
            if (bit != '0' and bit != '1') return error.InvalidBitCharacter;
            if (bit != '0') return error.InvalidEncoding;
        } else if (bit == '0') {
            current = current.left orelse return error.InvalidEncoding;
        } else if (bit == '1') {
            current = current.right orelse return error.InvalidEncoding;
        } else {
            return error.InvalidBitCharacter;
        }

        if (current.isLeaf()) {
            result.appendAssumeCapacity(current.symbol orelse return error.InvalidTree);
            current = root; // Reset to root for next symbol
        }
    }

    if (current != root) {
        return error.IncompleteEncoding;
    }

    return result.toOwnedSlice(gpa);
}

/// Destroys Huffman tree and frees all nodes
///
/// Iterative: rotates each left child above its parent until the tree is a right-leaning list,
/// freeing nodes as the list is walked, so it needs neither recursion nor extra memory. The
/// tree must have at most `nodes_max` nodes, as every tree built from bytes does.
///
/// Time: O(n) — each node is rotated at most once and freed once
/// Space: O(1)
pub fn destroyHuffmanTree(gpa: Allocator, root: *HuffmanNode) void {
    var current: ?*HuffmanNode = root;
    for (0..2 * nodes_max) |_| {
        const node = current orelse break;
        if (node.left) |left| {
            node.left = left.right;
            left.right = node;
            current = left;
        } else {
            current = node.right;
            gpa.destroy(node);
        }
    }
    assert(current == null);
}

fn freeCodes(gpa: Allocator, codes: *ArrayList(HuffmanCode)) void {
    for (codes.items) |code| gpa.free(code.code);
    codes.deinit(gpa);
}

test "huffman - basic encoding/decoding" {
    const data = "aabbbcccc";

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer freeCodes(testing.allocator, &codes);

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
    defer freeCodes(testing.allocator, &codes);

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
    defer freeCodes(testing.allocator, &codes);

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
    defer freeCodes(testing.allocator, &codes);

    try testing.expectEqual(@as(usize, 6), codes.items.len);

    const encoded = try huffmanEncode(testing.allocator, data, codes.items);
    defer testing.allocator.free(encoded);

    const decoded = try huffmanDecode(testing.allocator, encoded, tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualStrings(data, decoded);
}

test "huffman - two symbols equal frequency get length-1 codes" {
    const data = "ab";

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer freeCodes(testing.allocator, &codes);

    try testing.expectEqual(@as(usize, 2), codes.items.len);
    for (codes.items) |code| {
        try testing.expectEqual(@as(usize, 1), code.length);
    }
}

test "huffman - single byte input roundtrip" {
    const data = "x";

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer freeCodes(testing.allocator, &codes);

    const encoded = try huffmanEncode(testing.allocator, data, codes.items);
    defer testing.allocator.free(encoded);

    const decoded = try huffmanDecode(testing.allocator, encoded, tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualStrings(data, decoded);
}

test "huffman - frequent symbol gets shorter or equal code" {
    const data = "aaaaaab"; // 6 a's, 1 b

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer freeCodes(testing.allocator, &codes);

    var a_len: usize = 0;
    var b_len: usize = 0;
    for (codes.items) |code| {
        if (code.symbol == 'a') a_len = code.length;
        if (code.symbol == 'b') b_len = code.length;
    }

    try testing.expect(a_len <= b_len);
}

test "huffman - decode empty encoded string returns empty" {
    const data = "ab";

    const tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, tree);

    const decoded = try huffmanDecode(testing.allocator, "", tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqual(@as(usize, 0), decoded.len);
}

test "huffman - memory safety loop" {
    for (0..10) |_| {
        const data = "hello world";

        const tree = try buildHuffmanTree(testing.allocator, data);
        defer destroyHuffmanTree(testing.allocator, tree);

        var codes = try generateHuffmanCodes(testing.allocator, tree);
        defer freeCodes(testing.allocator, &codes);

        const encoded = try huffmanEncode(testing.allocator, data, codes.items);
        defer testing.allocator.free(encoded);

        const decoded = try huffmanDecode(testing.allocator, encoded, tree);
        defer testing.allocator.free(decoded);

        try testing.expectEqualStrings(data, decoded);
    }
}

test "huffman - single character stream decodes one symbol per bit" {
    const tree = try buildHuffmanTree(testing.allocator, "zzz");
    defer destroyHuffmanTree(testing.allocator, tree);

    const decoded = try huffmanDecode(testing.allocator, "000", tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualStrings("zzz", decoded);
    try testing.expectError(error.InvalidEncoding, huffmanDecode(testing.allocator, "01", tree));
    try testing.expectError(error.InvalidBitCharacter, huffmanDecode(testing.allocator, "0x", tree));
}

test "huffman - decode rejects malformed streams" {
    const tree = try buildHuffmanTree(testing.allocator, "aabbbcccc");
    defer destroyHuffmanTree(testing.allocator, tree);

    try testing.expectError(error.InvalidBitCharacter, huffmanDecode(testing.allocator, "2", tree));
    // A code that stops inside the tree is incomplete: every internal node needs both bits.
    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer freeCodes(testing.allocator, &codes);

    var longest: HuffmanCode = codes.items[0];
    for (codes.items) |code| {
        if (code.length > longest.length) longest = code;
    }
    try testing.expect(longest.length >= 2);
    const truncated = longest.code[0 .. longest.length - 1];
    try testing.expectError(
        error.IncompleteEncoding,
        huffmanDecode(testing.allocator, truncated, tree),
    );
}

test "huffman - encode reports a symbol without a code" {
    const tree = try buildHuffmanTree(testing.allocator, "ab");
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer freeCodes(testing.allocator, &codes);

    try testing.expectError(error.SymbolNotFound, huffmanEncode(testing.allocator, "abc", codes.items));
}

test "huffman - tree for every byte value stays within the limits" {
    // Fibonacci-like frequencies make the deepest possible tree: 256 symbols, depth 255.
    var data: [256]u8 = undefined;
    for (&data, 0..) |*byte, i| byte.* = @intCast(i);

    const tree = try buildHuffmanTree(testing.allocator, &data);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer freeCodes(testing.allocator, &codes);

    try testing.expectEqual(@as(usize, 256), codes.items.len);
    for (codes.items) |code| try testing.expectEqual(@as(usize, 8), code.length);

    const encoded = try huffmanEncode(testing.allocator, &data, codes.items);
    defer testing.allocator.free(encoded);

    const decoded = try huffmanDecode(testing.allocator, encoded, tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, &data, decoded);
}

test "huffman - skewed frequencies build a deep chain" {
    // Symbol i appears 2^i times for i in 0..20, so the tree degenerates into a chain.
    var data: ArrayList(u8) = .empty;
    defer data.deinit(testing.allocator);

    for (0..20) |i| {
        try data.appendNTimes(testing.allocator, @intCast('a' + i), @as(usize, 1) << @intCast(i));
    }

    const tree = try buildHuffmanTree(testing.allocator, data.items);
    defer destroyHuffmanTree(testing.allocator, tree);

    var codes = try generateHuffmanCodes(testing.allocator, tree);
    defer freeCodes(testing.allocator, &codes);

    var longest: usize = 0;
    for (codes.items) |code| longest = @max(longest, code.length);
    try testing.expectEqual(@as(usize, 19), longest);

    const encoded = try huffmanEncode(testing.allocator, data.items, codes.items);
    defer testing.allocator.free(encoded);

    const decoded = try huffmanDecode(testing.allocator, encoded, tree);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, data.items, decoded);
}

test "huffman - same data builds the same codes" {
    const data = "the quick brown fox jumps over the lazy dog";

    var first_tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, first_tree);

    var second_tree = try buildHuffmanTree(testing.allocator, data);
    defer destroyHuffmanTree(testing.allocator, second_tree);

    var first = try generateHuffmanCodes(testing.allocator, first_tree);
    defer freeCodes(testing.allocator, &first);

    var second = try generateHuffmanCodes(testing.allocator, second_tree);
    defer freeCodes(testing.allocator, &second);

    try testing.expectEqual(first.items.len, second.items.len);
    for (first.items, second.items) |a, b| {
        try testing.expectEqual(a.symbol, b.symbol);
        try testing.expectEqualStrings(a.code, b.code);
    }
}

test "huffman - no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const data = "aabbbcccc";
            const tree = try buildHuffmanTree(gpa, data);
            defer destroyHuffmanTree(gpa, tree);

            var codes = try generateHuffmanCodes(gpa, tree);
            defer freeCodes(gpa, &codes);

            const encoded = try huffmanEncode(gpa, data, codes.items);
            defer gpa.free(encoded);

            const decoded = try huffmanDecode(gpa, encoded, tree);
            defer gpa.free(decoded);

            try testing.expectEqualStrings(data, decoded);
        }
    };

    try testing.checkAllAllocationFailures(testing.allocator, Runner.run, .{});
}
