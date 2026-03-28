//! Online Algorithms - Process input piece-by-piece without full knowledge
//!
//! Online algorithms make irrevocable decisions on partial information,
//! evaluated via competitive analysis: ratio of online cost to optimal offline cost.
//!
//! ## Modules
//!
//! ### Ski Rental (`ski_rental`)
//! Classic rent-vs-buy decision problem under uncertainty.
//! - **Deterministic Strategy**: 2-competitive (buy after b/r days)
//! - **Randomized Strategy**: e/(e-1) ≈ 1.58-competitive (exponential distribution)
//! - **Applications**: Cloud resource management, equipment leasing, caching
//!
//! ### Paging (`paging`)
//! Page replacement for limited memory capacity.
//! - **LRU**: k-competitive, O(1) per request
//! - **FIFO**: k-competitive, O(1) per request
//! - **MIN (offline optimal)**: Perfect knowledge of future requests
//! - **Applications**: Virtual memory, database buffers, web caching, CDNs
//!
//! ### Load Balancing (`load_balancing`)
//! Assign jobs to machines to minimize makespan (max load).
//! - **Greedy**: (2 - 1/m)-competitive, assigns to least loaded machine
//! - **Round Robin**: Simple cyclic assignment, ignores load
//! - **Applications**: Task scheduling, cloud computing, web servers, parallel processing
//!
//! ### Bipartite Matching (`bipartite_matching`)
//! Match online vertices to offline vertices immediately and irrevocably.
//! - **Greedy**: 1/2-competitive, matches to any available neighbor
//! - **Ranking**: (1 - 1/e) ≈ 0.632-competitive, matches to highest-ranked neighbor
//! - **Applications**: Online advertising, job assignment, resource allocation, ridesharing
//!
//! ## Algorithm Selection Guide
//!
//! | Problem                     | Algorithm          | Competitive Ratio | Complexity     |
//! |-----------------------------|--------------------|--------------------|----------------|
//! | Rent vs. buy decision       | Ski Rental (det)   | 2.0                | O(1)           |
//! | Rent vs. buy decision       | Ski Rental (rand)  | ~1.58              | O(1)           |
//! | Page replacement (recency)  | LRU Paging         | k                  | O(1)           |
//! | Page replacement (simple)   | FIFO Paging        | k                  | O(1)           |
//! | Job scheduling (load-aware) | Greedy Load Bal    | 2 - 1/m            | O(m)           |
//! | Job scheduling (simple)     | Round Robin        | No guarantee       | O(1)           |
//! | Bipartite matching          | Greedy Matching    | 1/2                | O(k)           |
//! | Bipartite matching          | Ranking Matching   | ~0.632             | O(k)           |
//!
//! ## Competitive Analysis Basics
//!
//! - **Competitive Ratio**: max(online_cost / offline_cost) over all inputs
//! - **c-competitive**: Online algorithm achieves at most c × optimal offline cost
//! - **Deterministic**: Fixed strategy, worst-case analysis
//! - **Randomized**: Uses randomness, expected competitive ratio
//! - **Lower bounds**: No algorithm can do better than certain ratio
//!
//! ## Common Patterns
//!
//! 1. **Greedy strategies**: Make locally optimal decisions (simple, often 2-competitive)
//! 2. **Randomization**: Use random choices to improve competitive ratio
//! 3. **Potential functions**: Prove competitive ratios via amortized analysis
//! 4. **Work functions**: Track cost difference between online and offline
//!
//! ## Examples
//!
//! ```zig
//! const ski_rental = @import("zuda").algorithms.online.ski_rental;
//! const paging = @import("zuda").algorithms.online.paging;
//! const load_balancing = @import("zuda").algorithms.online.load_balancing;
//! const bipartite_matching = @import("zuda").algorithms.online.bipartite_matching;
//!
//! // Ski rental: deterministic strategy
//! const problem = ski_rental.Problem{ .rent_cost_per_day = 10.0, .buy_cost = 100.0 };
//! var strategy = ski_rental.DeterministicStrategy.init(problem);
//! for (0..days) |_| {
//!     const decision = strategy.nextDay();
//!     if (decision == .buy) break;
//! }
//!
//! // Paging: LRU cache
//! var lru = try paging.LRU(u32).init(allocator, 100);
//! defer lru.deinit();
//! const is_fault = try lru.request(page_id);
//! const stats = lru.getStats();
//! std.debug.print("Hit rate: {d:.2}%\n", .{stats.hitRate() * 100});
//!
//! // Load balancing: greedy assignment
//! var balancer = try load_balancing.GreedyLoadBalancer.init(allocator, 10);
//! defer balancer.deinit();
//! const machine_id = try balancer.assignJob(.{ .id = 1, .load = 5.0 });
//! const makespan = balancer.getMakespan();
//!
//! // Bipartite matching: ranking algorithm
//! var matcher = try bipartite_matching.RankingMatcher.init(allocator, 50, random);
//! defer matcher.deinit();
//! const match = try matcher.matchOnlineVertex(vertex_id, neighbors);
//! ```

pub const ski_rental = @import("online/ski_rental.zig");
pub const paging = @import("online/paging.zig");
pub const load_balancing = @import("online/load_balancing.zig");
pub const bipartite_matching = @import("online/bipartite_matching.zig");
