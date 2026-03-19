# Migrating zr Graph Algorithms to zuda

## Overview

This guide shows how to migrate zr's custom graph implementations to zuda's graph data structures and algorithms.

**Impact**: -715 LOC from zr (DAG 187 + TopoSort 323 + CycleDetect 205)

**Repositories**:
- `../zr/src/graph/dag.zig` (187 LOC)
- `../zr/src/graph/topo_sort.zig` (323 LOC)
- `../zr/src/graph/cycle_detect.zig` (205 LOC)

---

## API Comparison

### zr DAG API (Current)

```zig
const DAG = struct {
    nodes: std.StringHashMap(Node),
    allocator: std.mem.Allocator,

    const Node = struct {
        id: []const u8,
        dependencies: std.ArrayList([]const u8),  // Edges
    };

    pub fn init(allocator: std.mem.Allocator) DAG;
    pub fn deinit(self: *DAG) void;
    pub fn addNode(self: *DAG, id: []const u8) !void;
    pub fn addEdge(self: *DAG, from: []const u8, to: []const u8) !void;
    pub fn topologicalSort(self: *DAG) ![][]const u8;  // Kahn's algorithm
    pub fn detectCycle(self: *DAG) !?[][]const u8;  // DFS-based
};
```

### zuda Graph API (Target)

```zig
const zuda = @import("zuda");
const std = @import("std");

// Graph representation (choose based on use case)
const Graph = zuda.containers.graphs.AdjacencyList([]const u8, void, .directed);

// Algorithms (separate from representation)
const topoSort = zuda.algorithms.graph.topologicalSort;
const cycleDetect = zuda.algorithms.graph.cycleDetection;

// Example usage
var graph = try Graph.init(allocator);
defer graph.deinit();

try graph.addVertex("task1");
try graph.addVertex("task2");
try graph.addEdge("task1", "task2", {});  // task1 → task2

// Topological sort (Kahn's algorithm)
const sorted = try topoSort.kahn(allocator, &graph);
defer allocator.free(sorted);

// Cycle detection (DFS-based)
const cycle = try cycleDetect.dfs(allocator, &graph);
defer if (cycle) |c| allocator.free(c);
```

---

## Key Differences

| Feature | zr DAG | zuda Graph | Notes |
|---------|--------|------------|-------|
| Representation | Embedded (HashMap of nodes) | Separate (AdjacencyList) | zuda: More flexible, supports multiple graph types |
| Algorithms | Embedded (methods on DAG) | Standalone functions | zuda: Algorithms work on any graph representation |
| Vertex Type | `[]const u8` only | `comptime K: type` | zuda: Generic (supports integers, structs, etc.) |
| Edge Data | None | `comptime V: type` | zuda: Can attach weights, labels, etc. |
| Cycle Detection | DFS only | DFS + Union-Find | zuda: Multiple algorithms |
| Topological Sort | Kahn only | Kahn + DFS | zuda: Multiple algorithms |
| Error Handling | Returns `error.CycleDetected` | Returns cycle path | zuda: More actionable errors |

---

## Migration Strategy

### Phase 1: Compatibility Wrapper (Low Risk)

Create a wrapper that preserves zr's existing DAG API:

```zig
// zr/src/graph/dag.zig (new implementation)
const zuda = @import("zuda");
const std = @import("std");

const Graph = zuda.containers.graphs.AdjacencyList([]const u8, void, .directed);
const topoSort = zuda.algorithms.graph.topologicalSort;
const cycleDetect = zuda.algorithms.graph.cycleDetection;

pub const DAG = struct {
    graph: Graph,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) DAG {
        return .{
            .graph = Graph.init(allocator) catch unreachable,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DAG) void {
        self.graph.deinit();
    }

    pub fn addNode(self: *DAG, id: []const u8) !void {
        try self.graph.addVertex(id);
    }

    pub fn addEdge(self: *DAG, from: []const u8, to: []const u8) !void {
        try self.graph.addEdge(from, to, {});
    }

    pub fn topologicalSort(self: *DAG) ![][]const u8 {
        // Use Kahn's algorithm (matches old behavior)
        return topoSort.kahn(self.allocator, &self.graph);
    }

    pub fn detectCycle(self: *DAG) !?[][]const u8 {
        // Use DFS cycle detection (matches old behavior)
        return cycleDetect.dfs(self.allocator, &self.graph);
    }
};
```

**Testing**: Run zr's existing graph test suite. All tests should pass without modification.

### Phase 2: Direct Usage (High Reward)

Once confidence is established, use zuda graph APIs directly:

```zig
// zr task scheduler
const zuda = @import("zuda");
const Graph = zuda.containers.graphs.AdjacencyList([]const u8, void, .directed);
const topoSort = zuda.algorithms.graph.topologicalSort;

pub const Scheduler = struct {
    task_graph: Graph,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Scheduler {
        return .{
            .task_graph = try Graph.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn addTask(self: *Scheduler, task: []const u8, dependencies: []const []const u8) !void {
        try self.task_graph.addVertex(task);
        for (dependencies) |dep| {
            try self.task_graph.addEdge(dep, task, {});  // dep → task
        }
    }

    pub fn executionOrder(self: *Scheduler) ![][]const u8 {
        // Use zuda topological sort directly
        const sorted = try topoSort.kahn(self.allocator, &self.task_graph);

        // Check for cycles (Kahn returns error.CycleDetected if cycle exists)
        errdefer self.allocator.free(sorted);

        return sorted;
    }
};
```

---

## Performance Validation

### Benchmark Setup

```zig
// bench/graph_migration.zig
const std = @import("std");
const zuda = @import("zuda");
const zr_old = @import("zr_old");

const BenchContext = struct {
    allocator: std.mem.Allocator,
    num_nodes: usize,
    edges_per_node: usize,
};

fn benchTopoSort(comptime GraphType: type, ctx: BenchContext) !u64 {
    var graph = GraphType.init(ctx.allocator);
    defer graph.deinit();

    // Build random DAG
    for (0..ctx.num_nodes) |i| {
        const node = try std.fmt.allocPrint(ctx.allocator, "node_{}", .{i});
        try graph.addNode(node);
        if (i > 0) {
            for (0..@min(ctx.edges_per_node, i)) |j| {
                const dep = try std.fmt.allocPrint(ctx.allocator, "node_{}", .{j});
                try graph.addEdge(dep, node);
            }
        }
    }

    const start = std.time.nanoTimestamp();
    const sorted = try graph.topologicalSort();
    defer ctx.allocator.free(sorted);
    const end = std.time.nanoTimestamp();

    return @intCast(end - start);
}

pub fn main() !void {
    const ctx = BenchContext{ .allocator = allocator, .num_nodes = 10000, .edges_per_node = 5 };

    const old_ns = try benchTopoSort(zr_old.DAG, ctx);
    const new_ns = try benchTopoSort(ZudaGraph, ctx);

    std.debug.print("zr (old):   {} µs\n", .{old_ns / 1000});
    std.debug.print("zuda (new): {} µs\n", .{new_ns / 1000});
}
```

### Expected Results

| Operation | zr (old) | zuda (new) | Notes |
|-----------|----------|------------|-------|
| Topological sort (10k nodes) | ~80 µs | ~70 µs | zuda: Better cache locality |
| Cycle detection (10k nodes) | ~120 µs | ~100 µs | zuda: Optimized DFS |
| Graph construction (10k nodes) | ~50 µs | ~45 µs | zuda: Allocation batching |

---

## Memory Footprint

### zr DAG (Current)

```
StringHashMap(Node): ~32 bytes/entry overhead
Node: ~48 bytes (id pointer + ArrayList)
Dependencies: 8 bytes/edge

10k nodes, 50k edges:
- Nodes: 10k × 80 bytes = 800 KB
- Edges: 50k × 8 bytes = 400 KB
Total: ~1.2 MB
```

### zuda AdjacencyList (Target)

```
HashMap(K, ArrayList(K)): ~24 bytes/entry overhead
Edges stored in ArrayList: 8 bytes/edge

10k nodes, 50k edges:
- Nodes: 10k × 24 bytes = 240 KB
- Edges: 50k × 8 bytes = 400 KB
Total: ~640 KB (47% reduction!)
```

**Conclusion**: zuda uses less memory due to more compact representation.

---

## Advanced Features

### zuda Provides Additional Algorithms

After migration, zr gains access to:

1. **Multiple topological sort algorithms**:
   ```zig
   // Kahn's algorithm (BFS-based, original behavior)
   const sorted1 = try topoSort.kahn(allocator, &graph);

   // DFS-based (sometimes faster for sparse graphs)
   const sorted2 = try topoSort.dfs(allocator, &graph);
   ```

2. **Enhanced cycle detection**:
   ```zig
   // DFS-based (returns full cycle path)
   const cycle = try cycleDetect.dfs(allocator, &graph);
   if (cycle) |path| {
       std.debug.print("Cycle detected: {any}\n", .{path});
   }
   ```

3. **Graph analysis**:
   ```zig
   // Strongly connected components (future feature)
   const scc = try zuda.algorithms.graph.stronglyConnectedComponents(allocator, &graph);
   ```

---

## Migration Checklist

- [ ] Add zuda dependency to zr's `build.zig.zon`
- [ ] Create compatibility wrapper in `src/graph/dag.zig`
- [ ] Run full zr test suite — verify 0 failures
- [ ] Benchmark old vs new implementation (topo sort, cycle detection)
- [ ] Document any performance changes in zr CHANGELOG
- [ ] Create PR with migration (link back to this guide)
- [ ] After merge: Delete old implementations (715 LOC total)
  - `src/graph/dag.zig` (187 LOC)
  - `src/graph/topo_sort.zig` (323 LOC)
  - `src/graph/cycle_detect.zig` (205 LOC)
- [ ] Update zr docs to reference zuda graph documentation

---

## Rollback Plan

If critical issues arise:

1. **Git revert** — Single commit migration makes rollback trivial
2. **Keep old implementation** — Archive old graph/ directory as graph_legacy/
3. **Report to zuda** — Open issue with reproduction case

---

## References

- zuda AdjacencyList: `zuda/src/containers/graphs/adjacency_list.zig`
- zuda Topological Sort: `zuda/src/algorithms/graph/topological_sort.zig`
- zuda Cycle Detection: `zuda/src/algorithms/graph/cycle_detection.zig`
- zuda Graph tests: 30+ tests covering correctness, edge cases, memory safety
- zr current implementations:
  - `zr/src/graph/dag.zig` (187 LOC)
  - `zr/src/graph/topo_sort.zig` (323 LOC)
  - `zr/src/graph/cycle_detect.zig` (205 LOC)

---

## Contact

Questions about this migration? Open an issue on zuda or zr repositories.
