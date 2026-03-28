//! Online Bipartite Matching - Match items as they arrive
//!
//! Problem: Given a bipartite graph where one side (offline vertices) is known
//! in advance and the other side (online vertices) arrives one at a time.
//! Each online vertex must be matched immediately and irrevocably.
//!
//! Competitive Analysis:
//! - Greedy: 1/2-competitive (worst case)
//! - Ranking Algorithm: (1 - 1/e) ≈ 0.632-competitive
//!
//! Applications:
//! - Online advertising (ads to users)
//! - Job assignment (workers to tasks)
//! - Resource allocation in real-time systems
//! - Ridesharing (drivers to passengers)

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Match between online and offline vertices
pub const Match = struct {
    online_vertex: usize,
    offline_vertex: usize,
};

/// Greedy Online Bipartite Matching
/// Matches each online vertex to any available neighbor
/// Time: O(n * m) where n is online vertices, m is offline vertices | Space: O(n + m)
pub const GreedyMatcher = struct {
    allocator: Allocator,
    num_offline: usize,
    matched_offline: std.DynamicBitSet, // Which offline vertices are matched
    matches: std.ArrayList(Match),

    /// Initialize with number of offline vertices
    /// Time: O(m) | Space: O(m)
    pub fn init(allocator: Allocator, num_offline: usize) !GreedyMatcher {
        return GreedyMatcher{
            .allocator = allocator,
            .num_offline = num_offline,
            .matched_offline = try std.DynamicBitSet.initEmpty(allocator, num_offline),
            .matches = std.ArrayList(Match).init(allocator),
        };
    }

    /// Clean up resources
    pub fn deinit(self: *GreedyMatcher) void {
        self.matched_offline.deinit();
        self.matches.deinit();
    }

    /// Process online vertex with its neighbors
    /// Returns offline vertex matched to, or null if no match possible
    /// Time: O(k) where k is number of neighbors | Space: O(1)
    pub fn matchOnlineVertex(
        self: *GreedyMatcher,
        online_vertex: usize,
        neighbors: []const usize,
    ) !?usize {
        // Find first unmatched neighbor
        for (neighbors) |offline_vertex| {
            if (offline_vertex >= self.num_offline) continue;
            if (!self.matched_offline.isSet(offline_vertex)) {
                // Match found
                self.matched_offline.set(offline_vertex);
                try self.matches.append(.{
                    .online_vertex = online_vertex,
                    .offline_vertex = offline_vertex,
                });
                return offline_vertex;
            }
        }

        return null; // No match available
    }

    /// Get number of matches
    /// Time: O(1) | Space: O(1)
    pub fn matchCount(self: GreedyMatcher) usize {
        return self.matches.items.len;
    }

    /// Get all matches
    /// Time: O(1) | Space: O(1)
    pub fn getMatches(self: GreedyMatcher) []const Match {
        return self.matches.items;
    }

    /// Check if offline vertex is matched
    /// Time: O(1) | Space: O(1)
    pub fn isOfflineMatched(self: GreedyMatcher, offline_vertex: usize) bool {
        if (offline_vertex >= self.num_offline) return false;
        return self.matched_offline.isSet(offline_vertex);
    }
};

/// Ranking Algorithm for Online Bipartite Matching
/// Assigns random ranks to offline vertices, matches to highest-ranked available neighbor
/// Achieves (1 - 1/e) ≈ 0.632 competitive ratio
/// Time: O(n * (m + log m)) | Space: O(n + m)
pub const RankingMatcher = struct {
    allocator: Allocator,
    num_offline: usize,
    ranks: []f64, // Random ranks for offline vertices
    matched_offline: std.DynamicBitSet,
    matches: std.ArrayList(Match),

    /// Initialize with number of offline vertices and random generator
    /// Time: O(m) | Space: O(m)
    pub fn init(allocator: Allocator, num_offline: usize, random: std.Random) !RankingMatcher {
        const ranks = try allocator.alloc(f64, num_offline);
        errdefer allocator.free(ranks);

        // Assign random ranks uniformly in [0, 1]
        for (ranks) |*rank| {
            rank.* = random.float(f64);
        }

        return RankingMatcher{
            .allocator = allocator,
            .num_offline = num_offline,
            .ranks = ranks,
            .matched_offline = try std.DynamicBitSet.initEmpty(allocator, num_offline),
            .matches = std.ArrayList(Match).init(allocator),
        };
    }

    /// Clean up resources
    pub fn deinit(self: *RankingMatcher) void {
        self.allocator.free(self.ranks);
        self.matched_offline.deinit();
        self.matches.deinit();
    }

    /// Process online vertex with its neighbors
    /// Matches to highest-ranked unmatched neighbor
    /// Time: O(k) where k is number of neighbors | Space: O(1)
    pub fn matchOnlineVertex(
        self: *RankingMatcher,
        online_vertex: usize,
        neighbors: []const usize,
    ) !?usize {
        // Find unmatched neighbor with highest rank
        var best_offline: ?usize = null;
        var best_rank: f64 = -1.0;

        for (neighbors) |offline_vertex| {
            if (offline_vertex >= self.num_offline) continue;
            if (!self.matched_offline.isSet(offline_vertex)) {
                const rank = self.ranks[offline_vertex];
                if (rank > best_rank) {
                    best_rank = rank;
                    best_offline = offline_vertex;
                }
            }
        }

        if (best_offline) |offline_vertex| {
            self.matched_offline.set(offline_vertex);
            try self.matches.append(.{
                .online_vertex = online_vertex,
                .offline_vertex = offline_vertex,
            });
            return offline_vertex;
        }

        return null;
    }

    /// Get number of matches
    /// Time: O(1) | Space: O(1)
    pub fn matchCount(self: RankingMatcher) usize {
        return self.matches.items.len;
    }

    /// Get all matches
    /// Time: O(1) | Space: O(1)
    pub fn getMatches(self: RankingMatcher) []const Match {
        return self.matches.items;
    }

    /// Check if offline vertex is matched
    /// Time: O(1) | Space: O(1)
    pub fn isOfflineMatched(self: RankingMatcher, offline_vertex: usize) bool {
        if (offline_vertex >= self.num_offline) return false;
        return self.matched_offline.isSet(offline_vertex);
    }

    /// Get rank of offline vertex
    /// Time: O(1) | Space: O(1)
    pub fn getRank(self: RankingMatcher, offline_vertex: usize) f64 {
        if (offline_vertex >= self.num_offline) return 0.0;
        return self.ranks[offline_vertex];
    }
};

/// Compute maximum matching offline (for comparison)
/// Uses simple greedy on full graph
/// Time: O(n * m) | Space: O(m)
pub fn maxMatchingOffline(
    allocator: Allocator,
    num_offline: usize,
    online_vertices: []const []const usize,
) !usize {
    var matched = try std.DynamicBitSet.initEmpty(allocator, num_offline);
    defer matched.deinit();

    var count: usize = 0;

    for (online_vertices) |neighbors| {
        for (neighbors) |offline_vertex| {
            if (offline_vertex >= num_offline) continue;
            if (!matched.isSet(offline_vertex)) {
                matched.set(offline_vertex);
                count += 1;
                break;
            }
        }
    }

    return count;
}

/// Compute competitive ratio
/// Time: O(1) | Space: O(1)
pub fn competitiveRatio(online_matches: usize, offline_matches: usize) f64 {
    if (offline_matches == 0) return 1.0;
    return @as(f64, @floatFromInt(online_matches)) / @as(f64, @floatFromInt(offline_matches));
}

// ============================================================================
// Tests
// ============================================================================

test "bipartite matching - greedy: basic matching" {
    var matcher = try GreedyMatcher.init(testing.allocator, 3);
    defer matcher.deinit();

    // Online vertex 0 → neighbors [0, 1]
    const match0 = try matcher.matchOnlineVertex(0, &[_]usize{ 0, 1 });
    try testing.expectEqual(@as(?usize, 0), match0);

    // Online vertex 1 → neighbors [0, 2]
    const match1 = try matcher.matchOnlineVertex(1, &[_]usize{ 0, 2 });
    try testing.expectEqual(@as(?usize, 2), match1); // 0 is taken

    // Online vertex 2 → neighbors [1]
    const match2 = try matcher.matchOnlineVertex(2, &[_]usize{1});
    try testing.expectEqual(@as(?usize, 1), match2);

    try testing.expectEqual(@as(usize, 3), matcher.matchCount());
}

test "bipartite matching - greedy: no available neighbors" {
    var matcher = try GreedyMatcher.init(testing.allocator, 2);
    defer matcher.deinit();

    _ = try matcher.matchOnlineVertex(0, &[_]usize{0});
    _ = try matcher.matchOnlineVertex(1, &[_]usize{1});

    // Online vertex 2 has neighbors already matched
    const match = try matcher.matchOnlineVertex(2, &[_]usize{ 0, 1 });
    try testing.expectEqual(@as(?usize, null), match);

    try testing.expectEqual(@as(usize, 2), matcher.matchCount());
}

test "bipartite matching - greedy: first-fit strategy" {
    var matcher = try GreedyMatcher.init(testing.allocator, 3);
    defer matcher.deinit();

    // Should match to first available neighbor
    const match = try matcher.matchOnlineVertex(0, &[_]usize{ 2, 1, 0 });
    try testing.expectEqual(@as(?usize, 2), match); // First in list
}

test "bipartite matching - greedy: offline status" {
    var matcher = try GreedyMatcher.init(testing.allocator, 3);
    defer matcher.deinit();

    try testing.expect(!matcher.isOfflineMatched(0));

    _ = try matcher.matchOnlineVertex(0, &[_]usize{0});

    try testing.expect(matcher.isOfflineMatched(0));
    try testing.expect(!matcher.isOfflineMatched(1));
    try testing.expect(!matcher.isOfflineMatched(2));
}

test "bipartite matching - greedy: matches structure" {
    var matcher = try GreedyMatcher.init(testing.allocator, 2);
    defer matcher.deinit();

    _ = try matcher.matchOnlineVertex(5, &[_]usize{0});
    _ = try matcher.matchOnlineVertex(7, &[_]usize{1});

    const matches = matcher.getMatches();
    try testing.expectEqual(@as(usize, 2), matches.len);
    try testing.expectEqual(@as(usize, 5), matches[0].online_vertex);
    try testing.expectEqual(@as(usize, 0), matches[0].offline_vertex);
    try testing.expectEqual(@as(usize, 7), matches[1].online_vertex);
    try testing.expectEqual(@as(usize, 1), matches[1].offline_vertex);
}

test "bipartite matching - ranking: basic matching" {
    var prng = std.Random.DefaultPrng.init(12345);
    var matcher = try RankingMatcher.init(testing.allocator, 3, prng.random());
    defer matcher.deinit();

    // Matches to highest-ranked neighbor
    _ = try matcher.matchOnlineVertex(0, &[_]usize{ 0, 1, 2 });

    try testing.expectEqual(@as(usize, 1), matcher.matchCount());
}

test "bipartite matching - ranking: rank-based selection" {
    var prng = std.Random.DefaultPrng.init(54321);
    var matcher = try RankingMatcher.init(testing.allocator, 3, prng.random());
    defer matcher.deinit();

    // All neighbors available, should pick highest rank
    const neighbors = [_]usize{ 0, 1, 2 };
    const match = try matcher.matchOnlineVertex(0, &neighbors);

    try testing.expect(match != null);

    // Verify it picked the highest rank
    var max_rank: f64 = -1.0;
    var max_idx: usize = 0;
    for (neighbors) |i| {
        const rank = matcher.getRank(i);
        if (rank > max_rank) {
            max_rank = rank;
            max_idx = i;
        }
    }

    try testing.expectEqual(max_idx, match.?);
}

test "bipartite matching - ranking: respects matched status" {
    var prng = std.Random.DefaultPrng.init(99999);
    var matcher = try RankingMatcher.init(testing.allocator, 3, prng.random());
    defer matcher.deinit();

    // Match vertex 0 to some neighbor
    _ = try matcher.matchOnlineVertex(0, &[_]usize{ 0, 1 });

    // Vertex 1 should not match to already-matched neighbor
    const match = try matcher.matchOnlineVertex(1, &[_]usize{ 0, 1, 2 });
    try testing.expect(match != null);
    try testing.expect(match.? != 0 or !matcher.isOfflineMatched(0));
}

test "bipartite matching - ranking: random ranks" {
    var prng = std.Random.DefaultPrng.init(11111);
    var matcher = try RankingMatcher.init(testing.allocator, 5, prng.random());
    defer matcher.deinit();

    // Ranks should be in [0, 1]
    for (0..5) |i| {
        const rank = matcher.getRank(i);
        try testing.expect(rank >= 0.0);
        try testing.expect(rank <= 1.0);
    }
}

test "bipartite matching - offline maximum matching" {
    const online_vertices = [_][]const usize{
        &[_]usize{ 0, 1 },
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
    };

    const max_match = try maxMatchingOffline(testing.allocator, 3, &online_vertices);

    try testing.expectEqual(@as(usize, 3), max_match);
}

test "bipartite matching - competitive ratio: greedy" {
    const online_vertices = [_][]const usize{
        &[_]usize{ 0, 1 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 1, 2 },
    };

    var greedy = try GreedyMatcher.init(testing.allocator, 3);
    defer greedy.deinit();

    for (online_vertices, 0..) |neighbors, i| {
        _ = try greedy.matchOnlineVertex(i, neighbors);
    }

    const online_matches = greedy.matchCount();
    const offline_matches = try maxMatchingOffline(testing.allocator, 3, &online_vertices);

    const ratio = competitiveRatio(online_matches, offline_matches);

    // Greedy is at least 1/2-competitive
    try testing.expect(ratio >= 0.5);
    try testing.expect(ratio <= 1.0);
}

test "bipartite matching - competitive ratio: ranking better than greedy" {
    const online_vertices = [_][]const usize{
        &[_]usize{ 0, 1, 2 },
        &[_]usize{ 0, 1, 2 },
        &[_]usize{ 0, 1, 2 },
    };

    // Run multiple trials to compare
    var greedy_sum: usize = 0;
    var ranking_sum: usize = 0;

    var prng = std.Random.DefaultPrng.init(77777);

    for (0..10) |_| {
        var greedy = try GreedyMatcher.init(testing.allocator, 3);
        defer greedy.deinit();

        var ranking = try RankingMatcher.init(testing.allocator, 3, prng.random());
        defer ranking.deinit();

        for (online_vertices, 0..) |neighbors, i| {
            _ = try greedy.matchOnlineVertex(i, neighbors);
            _ = try ranking.matchOnlineVertex(i, neighbors);
        }

        greedy_sum += greedy.matchCount();
        ranking_sum += ranking.matchCount();
    }

    // Ranking should perform at least as well as greedy on average
    try testing.expect(ranking_sum >= greedy_sum);
}

test "bipartite matching - large instance" {
    var greedy = try GreedyMatcher.init(testing.allocator, 100);
    defer greedy.deinit();

    // Create neighbors for 100 online vertices
    for (0..100) |i| {
        const neighbors = [_]usize{ i % 100, (i + 1) % 100 };
        _ = try greedy.matchOnlineVertex(i, &neighbors);
    }

    try testing.expect(greedy.matchCount() > 0);
}

test "bipartite matching - memory safety" {
    var greedy = try GreedyMatcher.init(testing.allocator, 50);
    defer greedy.deinit();

    for (0..200) |i| {
        const neighbors = [_]usize{ i % 50, (i + 1) % 50, (i + 2) % 50 };
        _ = try greedy.matchOnlineVertex(i, &neighbors);
    }

    var prng = std.Random.DefaultPrng.init(88888);
    var ranking = try RankingMatcher.init(testing.allocator, 50, prng.random());
    defer ranking.deinit();

    for (0..200) |i| {
        const neighbors = [_]usize{ i % 50, (i + 1) % 50, (i + 2) % 50 };
        _ = try ranking.matchOnlineVertex(i, &neighbors);
    }
}
