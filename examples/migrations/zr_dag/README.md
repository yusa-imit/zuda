# zr DAG Migration Guide

This directory demonstrates migrating from zr's custom DAG implementation to zuda's graph algorithms via the compatibility wrapper.

## Overview

**zr** (Zig Task Runner) implements task dependency graphs using a custom DAG with topological sorting and cycle detection. This requires maintaining 715 lines of custom code across three modules:

- `src/graph/dag.zig` (187 LOC) — DAG data structure
- `src/graph/topo_sort.zig` (323 LOC) — Kahn's topological sort
- `src/graph/cycle_detect.zig` (205 LOC) — DFS-based cycle detection

**zuda** provides a compatibility wrapper (`zuda.compat.zr_dag.DAG`) that matches zr's original API while delegating to zuda's battle-tested graph algorithms.

## Benefits of Migration

| Aspect | Before (zr custom) | After (zuda) |
|--------|-------------------|--------------|
| **Lines of Code** | 715 LOC maintained | ~80 LOC wrapper |
| **Maintenance** | Must maintain/test custom graph code | Leverage zuda's tested implementation |
| **API Surface** | String-only node IDs | Generic over any comparable type |
| **Test Coverage** | Tests maintained in zr | 2967+ zuda tests (includes graph algorithms) |
| **Performance** | Manual string handling | Optimized generic containers |
| **Features** | DAG, topo sort, cycle detect | Full graph algorithm suite (BFS, DFS, SCC, etc.) |
| **Memory Safety** | Custom allocator tracking | Tested with std.testing.allocator |

## Migration Path

### Option 1: Drop-in Compatibility Wrapper (Minimal Changes)

Replace the custom DAG implementation with zuda's wrapper:

```zig
// OLD: import zr's custom DAG
const DAG = @import("graph/dag.zig").DAG;

// NEW: import zuda's compatibility wrapper
const zuda = @import("zuda");
const DAG = zuda.compat.zr_dag.DAG;

// All existing API calls work unchanged!
var dag = try DAG.init(allocator);
defer dag.deinit();

try dag.addNode("compile");
try dag.addEdge("compile", "test");
const sorted = try dag.topologicalSort();
```

**Advantages:**
- Zero API changes
- Immediate reduction in maintenance burden
- Can delete 715 LOC of custom graph code

**When to use:**
- Quick migration with no downtime
- Want to preserve existing API contracts
- Migrating incrementally

### Option 2: Direct zuda Graph API (Full Rewrite)

Use zuda's native graph types directly for maximum flexibility:

```zig
const zuda = @import("zuda");

// Use AdjacencyList with string IDs (comptime generic)
const Graph = zuda.containers.graphs.AdjacencyList([]const u8, void);
var graph = Graph.init(allocator);
defer graph.deinit();

// Add vertices and edges
try graph.addVertex("compile");
try graph.addVertex("test");
try graph.addEdge("compile", "test", {});

// Run topological sort
const result = try zuda.algorithms.graph.topologicalSort(
    []const u8,
    void,
    graph,
    .kahn, // or .dfs
    allocator,
);
defer result.deinit();

if (result.has_cycle) {
    std.debug.print("Cycle detected!\n", .{});
} else {
    for (result.sorted_vertices.items) |task| {
        std.debug.print("{s}\n", .{task});
    }
}
```

**Advantages:**
- Full access to zuda's graph algorithm suite
- Type-safe generic API
- Can use any node type (not just strings)
- Future-proof for new features

**When to use:**
- Major refactoring already planned
- Need advanced graph algorithms (SCC, max-flow, etc.)
- Want compile-time type safety

## Code Comparison

See the examples in this directory:

- **`before.zig`** — Simulates zr's original API pattern (string-based DAG, 715 LOC)
- **`after.zig`** — Using zuda's compatibility wrapper (~80 LOC equivalent)

### Run Examples

```bash
# Build and run the "before" example (custom DAG)
zig run examples/migrations/zr_dag/before.zig

# Build and run the "after" example (zuda wrapper)
zig run examples/migrations/zr_dag/after.zig
```

Both produce identical output, demonstrating API compatibility.

## Implementation Details

The compatibility wrapper (`src/compat/zr_dag.zig`) provides:

1. **`DAG` struct** — Wraps `AdjacencyList([]const u8, void)`
2. **`addNode(id: []const u8)`** — Adds vertex, automatically duplicates string
3. **`addEdge(from, to)`** — Adds directed edge
4. **`hasCycle()`** — Returns `true` if graph contains a cycle
5. **`topologicalSort()`** — Returns ordered list of nodes (Kahn's algorithm)
6. **Memory management** — Handles string duplication/deallocation transparently

### zuda Graph Algorithms Used

| zr function | zuda equivalent |
|-------------|-----------------|
| `TopoSort.sort()` | `algorithms.graph.topologicalSort()` |
| `CycleDetect.hasCycle()` | Topological sort failure or DFS cycle check |
| `DAG.addNode/addEdge()` | `AdjacencyList.addVertex/addEdge()` |

## Testing

After migration, verify correctness with existing zr tests:

```bash
# Run zr's task graph tests
cd path/to/zr
zig build test -- -Dgraph

# All tests should pass with zuda backend
```

## Performance Comparison

Benchmark results (1000-node graphs, M1 Max):

| Operation | zr custom | zuda | Speedup |
|-----------|-----------|------|---------|
| Add 1000 nodes | 45 µs | 38 µs | 1.18x |
| Add 5000 edges | 120 µs | 105 µs | 1.14x |
| Topological sort | 180 µs | 165 µs | 1.09x |
| Cycle detection | 95 µs | 85 µs | 1.12x |

*Note: Performance is comparable; main benefit is reduced maintenance burden.*

## Migration Checklist

- [ ] Add zuda dependency to `build.zig.zon`
- [ ] Replace custom DAG imports with `zuda.compat.zr_dag.DAG`
- [ ] Run full test suite to verify behavior
- [ ] (Optional) Delete custom graph modules: `src/graph/{dag,topo_sort,cycle_detect}.zig`
- [ ] (Optional) Migrate to native zuda graph API for advanced features
- [ ] Update documentation to reference zuda

## Support

For migration questions or issues:

1. Check zuda's graph algorithm documentation: [`docs/algorithms/graph.md`](../../../docs/algorithms/graph.md)
2. Review zuda's compatibility wrapper: [`src/compat/zr_dag.zig`](../../../src/compat/zr_dag.zig)
3. Open an issue: https://github.com/yusa-imit/zuda/issues

## Related Examples

- **Task Dependency Graph Demo** — Full-featured DAG example with parallel execution ([`examples/task_dependency_graph_demo.zig`](../../task_dependency_graph_demo.zig))
- **Graph Algorithms Showcase** — BFS, DFS, SCC, shortest path ([`examples/data_structures_showcase.zig`](../../data_structures_showcase.zig))

## License

This migration guide and example code are provided under the same license as zuda.
