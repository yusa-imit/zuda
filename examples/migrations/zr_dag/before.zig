// BEFORE: Using zr's custom DAG implementation (715 LOC total)
//
// This simulates zr's original API pattern:
// - DAG: 187 LOC (src/graph/dag.zig)
// - TopoSort: 323 LOC (src/graph/topo_sort.zig)
// - CycleDetect: 205 LOC (src/graph/cycle_detect.zig)
// - String-based node IDs
// - Manual memory management

const std = @import("std");

// Simplified representation of zr's DAG API
const DAG = struct {
    allocator: std.mem.Allocator,
    nodes: std.StringHashMap(void),
    edges: std.StringHashMap(std.ArrayList([]const u8)),

    pub fn init(allocator: std.mem.Allocator) !DAG {
        return .{
            .allocator = allocator,
            .nodes = std.StringHashMap(void).init(allocator),
            .edges = std.StringHashMap(std.ArrayList([]const u8)).init(allocator),
        };
    }

    pub fn deinit(self: *DAG) void {
        // Free all duplicated strings
        var edge_iter = self.edges.valueIterator();
        while (edge_iter.next()) |list| {
            for (list.items) |node| {
                self.allocator.free(node);
            }
            list.deinit(self.allocator);
        }
        self.edges.deinit();

        var node_iter = self.nodes.keyIterator();
        while (node_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.nodes.deinit();
    }

    pub fn addNode(self: *DAG, id: []const u8) !void {
        const owned = try self.allocator.dupe(u8, id);
        try self.nodes.put(owned, {});
        const list = std.ArrayList([]const u8){};
        try self.edges.put(owned, list);
    }

    pub fn addEdge(self: *DAG, from: []const u8, to: []const u8) !void {
        const owned_to = try self.allocator.dupe(u8, to);
        const list = self.edges.getPtr(from) orelse return error.NodeNotFound;
        try list.append(self.allocator, owned_to);
    }

    pub fn hasCycle(self: *DAG) !bool {
        // Would implement DFS-based cycle detection
        _ = self;
        return false;
    }

    pub fn topologicalSort(self: *DAG) !std.ArrayList([]const u8) {
        // Would implement Kahn's algorithm
        var result = std.ArrayList([]const u8){};
        var iter = self.nodes.keyIterator();
        while (iter.next()) |key| {
            try result.append(self.allocator, key.*);
        }
        return result;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== BEFORE: zr Custom DAG (715 LOC total) ===\n", .{});

    var dag = try DAG.init(allocator);
    defer dag.deinit();

    // Build task dependency graph
    try dag.addNode("compile");
    try dag.addNode("test");
    try dag.addNode("build");
    try dag.addNode("deploy");

    try dag.addEdge("compile", "test");
    try dag.addEdge("test", "build");
    try dag.addEdge("build", "deploy");

    // Check for cycles
    const has_cycle = try dag.hasCycle();
    std.debug.print("Graph has cycle: {}\n", .{has_cycle});

    // Topological sort
    var sorted = try dag.topologicalSort();
    defer sorted.deinit(allocator);

    std.debug.print("Execution order: ", .{});
    for (sorted.items, 0..) |task, i| {
        if (i > 0) std.debug.print(" → ", .{});
        std.debug.print("{s}", .{task});
    }
    std.debug.print("\n\n", .{});

    std.debug.print("Issues with custom implementation:\n", .{});
    std.debug.print("  - 715 LOC spread across 3 files in zr\n", .{});
    std.debug.print("  - String-only node IDs (no generics)\n", .{});
    std.debug.print("  - Manual string duplication/ownership\n", .{});
    std.debug.print("  - Separate DAG/TopoSort/CycleDetect modules\n", .{});
    std.debug.print("  - Limited test coverage (basic cases only)\n", .{});
}
