// BEFORE: Using silica's custom B+Tree implementation (4,300 LOC)
//
// This simulates silica's original API pattern:
// - Runtime order configuration
// - String-only keys and values
// - Manual memory management
// - Duplicates all inserted data

const std = @import("std");

// Simplified representation of silica's BTree API
// (In reality, this would be 4,300 LOC in silica/src/storage/btree.zig)
const SilicaBTree = struct {
    order: u16,
    allocator: std.mem.Allocator,
    count: usize,
    // Internal implementation would go here (omitted for example)

    pub fn init(allocator: std.mem.Allocator, order: u16) !*SilicaBTree {
        const tree = try allocator.create(SilicaBTree);
        tree.* = .{
            .order = order,
            .allocator = allocator,
            .count = 0,
        };
        return tree;
    }

    pub fn deinit(self: *SilicaBTree) void {
        // Would free all internal nodes and data
        self.allocator.destroy(self);
    }

    pub fn insert(self: *SilicaBTree, key: []const u8, value: []const u8) !void {
        // Would duplicate key and value strings
        // Would navigate to correct leaf node
        // Would split nodes if necessary
        // For this example, just increment count
        _ = key;
        _ = value;
        self.count += 1;
    }

    pub fn get(self: *SilicaBTree, key: []const u8) ?[]const u8 {
        // Would traverse tree to find key
        _ = self;
        _ = key;
        return null;
    }

    pub fn remove(self: *SilicaBTree, key: []const u8) !bool {
        // Would traverse tree, remove key, merge nodes if needed
        _ = self;
        _ = key;
        return false;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== BEFORE: silica Custom B+Tree (4,300 LOC) ===\n", .{});

    // Create B+Tree with order 128 (runtime configuration)
    const tree = try SilicaBTree.init(allocator, 128);
    defer tree.deinit();

    // Insert key-value pairs
    const start = std.time.nanoTimestamp();

    try tree.insert("user:1001", "Alice");
    try tree.insert("user:1002", "Bob");
    try tree.insert("user:1003", "Charlie");

    const end = std.time.nanoTimestamp();
    const elapsed_ns = @as(u64, @intCast(end - start));

    // Lookup
    const value = tree.get("user:1002");

    std.debug.print("Inserted 3 entries in {} ns\n", .{elapsed_ns});
    std.debug.print("Lookup result: {?s}\n", .{value});
    std.debug.print("Tree count: {}\n", .{tree.count});

    // Remove
    const removed = try tree.remove("user:1001");
    std.debug.print("Removed 'user:1001': {}\n", .{removed});

    std.debug.print("\n", .{});
    std.debug.print("Issues with custom implementation:\n", .{});
    std.debug.print("  - 4,300 LOC to maintain in silica codebase\n", .{});
    std.debug.print("  - String-only API (no generics)\n", .{});
    std.debug.print("  - Runtime order configuration (overhead)\n", .{});
    std.debug.print("  - Manual string duplication logic\n", .{});
    std.debug.print("  - No property-based testing\n", .{});
}
