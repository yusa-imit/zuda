// BEFORE: Using zoltraak's custom Sorted Set implementation (1,800 LOC)
//
// This simulates zoltraak's original API pattern:
// - HashMap + sorted ArrayList hybrid
// - Redis ZADD/ZRANGE/ZRANK semantics
// - String members with float64 scores
// - O(n) insert/remove due to linear ArrayList operations

const std = @import("std");

// Simplified representation of zoltraak's SortedSet API
const SortedSet = struct {
    allocator: std.mem.Allocator,
    scores: std.StringHashMap(f64), // member → score lookup
    sorted_members: std.ArrayList(Member), // sorted by score for range queries

    const Member = struct {
        member: []const u8,
        score: f64,
    };

    pub fn init(allocator: std.mem.Allocator) SortedSet {
        return .{
            .allocator = allocator,
            .scores = std.StringHashMap(f64).init(allocator),
            .sorted_members = std.ArrayList(Member){},
        };
    }

    pub fn deinit(self: *SortedSet) void {
        var iter = self.scores.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.scores.deinit();

        for (self.sorted_members.items) |member| {
            self.allocator.free(member.member);
        }
        self.sorted_members.deinit(self.allocator);
    }

    pub fn add(self: *SortedSet, member: []const u8, member_score: f64) !void {
        const owned_member = try self.allocator.dupe(u8, member);

        // Insert into hash map
        try self.scores.put(owned_member, member_score);

        // Insert into sorted array (O(n) - find position + shift)
        var insert_index: usize = 0;
        for (self.sorted_members.items, 0..) |m, i| {
            if (member_score < m.score) {
                insert_index = i;
                break;
            }
            insert_index = i + 1;
        }
        try self.sorted_members.insert(self.allocator, insert_index, .{ .member = owned_member, .score = member_score });
    }

    pub fn score(self: *SortedSet, member: []const u8) ?f64 {
        return self.scores.get(member);
    }

    pub fn rank(self: *SortedSet, member: []const u8) ?usize {
        for (self.sorted_members.items, 0..) |m, i| {
            if (std.mem.eql(u8, m.member, member)) {
                return i;
            }
        }
        return null;
    }

    pub fn range(self: *SortedSet, start: usize, stop: usize) []const Member {
        if (start >= self.sorted_members.items.len) return &.{};
        const end = @min(stop + 1, self.sorted_members.items.len);
        return self.sorted_members.items[start..end];
    }

    pub fn count(self: *SortedSet) usize {
        return self.sorted_members.items.len;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== BEFORE: zoltraak Custom Sorted Set (1,800 LOC) ===\n", .{});

    var zset = SortedSet.init(allocator);
    defer zset.deinit();

    // Add players with scores (Redis ZADD)
    const start = std.time.nanoTimestamp();

    try zset.add("Alice", 95.5);
    try zset.add("Bob", 87.2);
    try zset.add("Charlie", 92.0);
    try zset.add("Diana", 99.0);

    const end = std.time.nanoTimestamp();
    const elapsed_ns = @as(u64, @intCast(end - start));

    std.debug.print("Added 4 members in {} ns\n", .{elapsed_ns});

    // Get score (Redis ZSCORE)
    if (zset.score("Bob")) |s| {
        std.debug.print("Bob's score: {d:.1}\n", .{s});
    }

    // Get rank (Redis ZRANK)
    if (zset.rank("Charlie")) |r| {
        std.debug.print("Charlie's rank: {}\n", .{r});
    }

    // Range query (Redis ZRANGE)
    const top3 = zset.range(0, 2);
    std.debug.print("Top 3: ", .{});
    for (top3, 0..) |m, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{s} ({d:.1})", .{ m.member, m.score });
    }
    std.debug.print("\n\n", .{});

    std.debug.print("Issues with custom implementation:\n", .{});
    std.debug.print("  - 1,800 LOC in zoltraak codebase\n", .{});
    std.debug.print("  - O(n) insert/remove (ArrayList shift)\n", .{});
    std.debug.print("  - String-only members (no generics)\n", .{});
    std.debug.print("  - Manual synchronization of HashMap + ArrayList\n", .{});
    std.debug.print("  - Limited test coverage\n", .{});
}
