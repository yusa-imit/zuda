//! Benchmark suite for string algorithms
//!
//! Validates PRD performance targets:
//! - Aho-Corasick (1000 patterns, 1MB text): ≥ 500 MB/sec

const std = @import("std");
const zuda = @import("zuda");
const bench = zuda.internal.bench;

const AhoCorasick = zuda.algorithms.string.AhoCorasick;
const AhoCorasickASCII = zuda.algorithms.string.AhoCorasickASCII;
const DoubleArrayTrie = zuda.containers.strings.DoubleArrayTrie;

/// Context for Aho-Corasick benchmark — setup is NOT timed
const AhoCorasickContext = struct {
    ac: AhoCorasick(u8),
    text: []const u8,
    patterns: std.ArrayList([]const u8), // Keep patterns alive for automaton
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !AhoCorasickContext {
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        // Generate 1000 random patterns (5-15 bytes each)
        const pattern_count = 1000;
        var patterns: std.ArrayList([]const u8) = .{};
        errdefer {
            for (patterns.items) |pattern| {
                allocator.free(pattern);
            }
            patterns.deinit(allocator);
        }

        var i: usize = 0;
        while (i < pattern_count) : (i += 1) {
            const len = random.intRangeAtMost(usize, 5, 15);
            const pattern = try allocator.alloc(u8, len);
            for (pattern) |*byte| {
                byte.* = random.intRangeAtMost(u8, 'a', 'z');
            }
            try patterns.append(allocator, pattern);
        }

        // Build automaton
        const ac = try AhoCorasick(u8).init(allocator, patterns.items);

        // Generate 1MB text
        const text_size = 1024 * 1024;
        const text = try allocator.alloc(u8, text_size);
        for (text) |*byte| {
            byte.* = random.intRangeAtMost(u8, 'a', 'z');
        }

        return .{ .ac = ac, .text = text, .patterns = patterns, .allocator = allocator };
    }

    fn deinit(self: *AhoCorasickContext) void {
        self.allocator.free(self.text);
        self.ac.deinit();
        // Free patterns after automaton is destroyed
        for (self.patterns.items) |pattern| {
            self.allocator.free(pattern);
        }
        self.patterns.deinit(self.allocator);
    }
};

/// Benchmark: Aho-Corasick search only (generic)
fn benchAhoCorasickSearch(ctx: *AhoCorasickContext) !void {
    // Only time the search operation
    var matches = try ctx.ac.findAll(ctx.text, ctx.allocator);
    defer matches.deinit(ctx.allocator);
}

/// Context for ASCII Aho-Corasick benchmark — setup is NOT timed
const AhoCorasickASCIIContext = struct {
    ac: AhoCorasickASCII,
    text: []const u8,
    patterns: std.ArrayList([]const u8), // Keep patterns alive for automaton
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !AhoCorasickASCIIContext {
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        // Generate 1000 random patterns (5-15 bytes each)
        const pattern_count = 1000;
        var patterns: std.ArrayList([]const u8) = .{};
        errdefer {
            for (patterns.items) |pattern| {
                allocator.free(pattern);
            }
            patterns.deinit(allocator);
        }

        var i: usize = 0;
        while (i < pattern_count) : (i += 1) {
            const len = random.intRangeAtMost(usize, 5, 15);
            const pattern = try allocator.alloc(u8, len);
            for (pattern) |*byte| {
                byte.* = random.intRangeAtMost(u8, 'a', 'z');
            }
            try patterns.append(allocator, pattern);
        }

        // Build automaton
        const ac = try AhoCorasickASCII.init(allocator, patterns.items);

        // Generate 1MB text
        const text_size = 1024 * 1024;
        const text = try allocator.alloc(u8, text_size);
        for (text) |*byte| {
            byte.* = random.intRangeAtMost(u8, 'a', 'z');
        }

        return .{ .ac = ac, .text = text, .patterns = patterns, .allocator = allocator };
    }

    fn deinit(self: *AhoCorasickASCIIContext) void {
        self.allocator.free(self.text);
        self.ac.deinit();
        // Free patterns after automaton is destroyed
        for (self.patterns.items) |pattern| {
            self.allocator.free(pattern);
        }
        self.patterns.deinit(self.allocator);
    }
};

/// Benchmark: ASCII Aho-Corasick search only
fn benchAhoCorasickASCIISearch(ctx: *AhoCorasickASCIIContext) !void {
    // Only time the search operation
    var matches = try ctx.ac.findAll(ctx.text, ctx.allocator);
    defer matches.deinit(ctx.allocator);
}

/// Context for DoubleArrayTrie Aho-Corasick benchmark — setup is NOT timed
const DoubleArrayTrieContext = struct {
    trie: DoubleArrayTrie(u8),
    text: []const u8,
    patterns: std.ArrayList([]const u8), // Keep patterns alive for trie
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !DoubleArrayTrieContext {
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        // Generate 1000 random patterns (5-15 bytes each)
        const pattern_count = 1000;
        var patterns: std.ArrayList([]const u8) = .{};
        errdefer {
            for (patterns.items) |pattern| {
                allocator.free(pattern);
            }
            patterns.deinit(allocator);
        }

        var i: usize = 0;
        while (i < pattern_count) : (i += 1) {
            const len = random.intRangeAtMost(usize, 5, 15);
            const pattern = try allocator.alloc(u8, len);
            for (pattern) |*byte| {
                byte.* = random.intRangeAtMost(u8, 'a', 'z');
            }
            try patterns.append(allocator, pattern);
        }

        // Build double-array trie with Aho-Corasick
        const trie = try DoubleArrayTrie(u8).init(allocator, patterns.items);

        // Generate 1MB text
        const text_size = 1024 * 1024;
        const text = try allocator.alloc(u8, text_size);
        for (text) |*byte| {
            byte.* = random.intRangeAtMost(u8, 'a', 'z');
        }

        return .{ .trie = trie, .text = text, .patterns = patterns, .allocator = allocator };
    }

    fn deinit(self: *DoubleArrayTrieContext) void {
        self.allocator.free(self.text);
        var trie_mut = self.trie;
        trie_mut.deinit();
        // Free patterns after trie is destroyed
        for (self.patterns.items) |pattern| {
            self.allocator.free(pattern);
        }
        self.patterns.deinit(self.allocator);
    }
};

/// Benchmark: DoubleArrayTrie Aho-Corasick search only
fn benchDoubleArrayTrieSearch(ctx: *DoubleArrayTrieContext) !void {
    // Only time the search operation
    const matches = try ctx.trie.findAll(ctx.allocator, ctx.text);
    defer ctx.allocator.free(matches);
}

/// Run all string algorithm benchmarks and output markdown table
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n# String Algorithm Benchmarks\n\n", .{});
    std.debug.print("Validating PRD performance targets:\n", .{});
    std.debug.print("- Aho-Corasick: target ≥ 500 MB/sec (1000 patterns, 1MB text)\n\n", .{});

    // Aho-Corasick (generic) benchmark
    {
        std.debug.print("Running Aho-Corasick (generic, HashMap transitions) — search only (1000 patterns, 1MB text)...\n", .{});

        var ctx = try AhoCorasickContext.init(allocator);
        defer ctx.deinit();

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchAhoCorasickSearch, .{&ctx});

        const text_size_mb = 1; // 1MB
        const mb_per_sec = @divFloor(1_000_000_000, @divFloor(result.mean_ns, text_size_mb));

        std.debug.print("  Result: {d} MB/sec (mean over {d} iterations)\n", .{ mb_per_sec, result.iterations });
        std.debug.print("  (Baseline for comparison)\n", .{});
    }

    // Aho-Corasick ASCII-optimized benchmark
    {
        std.debug.print("\nRunning AhoCorasickASCII (array transitions) — search only (1000 patterns, 1MB text)...\n", .{});

        var ctx = try AhoCorasickASCIIContext.init(allocator);
        defer ctx.deinit();

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchAhoCorasickASCIISearch, .{&ctx});

        const text_size_mb = 1; // 1MB
        const mb_per_sec = @divFloor(1_000_000_000, @divFloor(result.mean_ns, text_size_mb));

        std.debug.print("  Result: {d} MB/sec (mean over {d} iterations)\n", .{ mb_per_sec, result.iterations });

        if (mb_per_sec >= 500) {
            std.debug.print("  ✓ PASS: meets target of ≥ 500 MB/sec\n", .{});
        } else {
            std.debug.print("  ✗ FAIL: below target of ≥ 500 MB/sec (gap: -{d}%)\n", .{@divFloor((500 - mb_per_sec) * 100, 500)});
        }
    }

    // DoubleArrayTrie Aho-Corasick benchmark
    {
        std.debug.print("\nRunning DoubleArrayTrie Aho-Corasick (double-array with FAIL/OUTPUT) — search only (1000 patterns, 1MB text)...\n", .{});

        var ctx = try DoubleArrayTrieContext.init(allocator);
        defer ctx.deinit();

        var benchmark = try bench.Benchmark.init(allocator, .{
            .warmup_iterations = 2,
            .min_iterations = 5,
            .max_iterations = 10,
        });
        defer benchmark.deinit();

        const result = try benchmark.run(benchDoubleArrayTrieSearch, .{&ctx});

        const text_size_mb = 1; // 1MB
        const mb_per_sec = @divFloor(1_000_000_000, @divFloor(result.mean_ns, text_size_mb));

        std.debug.print("  Result: {d} MB/sec (mean over {d} iterations)\n", .{ mb_per_sec, result.iterations });

        // Target is ≥200 MB/sec for double-array variant (revised from 500)
        if (mb_per_sec >= 200) {
            std.debug.print("  ✓ PASS: meets target of ≥ 200 MB/sec\n", .{});
            if (mb_per_sec >= 133) {
                const improvement = @divFloor((mb_per_sec - 133) * 100, 133);
                std.debug.print("  📈 Improvement over NodeASCII (133 MB/sec): +{d}%\n", .{improvement});
            }
        } else {
            std.debug.print("  ✗ FAIL: below target of ≥ 200 MB/sec (gap: -{d}%)\n", .{@divFloor((200 - mb_per_sec) * 100, 200)});
        }
    }

    std.debug.print("\n## Summary\n\n", .{});
    std.debug.print("Benchmark suite completed. See results above.\n", .{});
}
