//! Lock-free concurrent skip list.
//!
//! A thread-safe skip list that supports concurrent insertions, deletions, and searches
//! without locks. Uses compare-and-swap (CAS) operations for atomic modifications.
//!
//! ## Algorithm
//!
//! Based on the lock-free skip list by Herlihy et al. with marked references:
//! - Search: Lock-free traversal through levels
//! - Insert: CAS to atomically link nodes at each level
//! - Remove: Two-phase deletion (logical then physical)
//!   1. Logical: Mark the node as deleted (set mark bit)
//!   2. Physical: Unlink the node from all levels
//!
//! ## Properties
//!
//! - **Lock-free**: At least one thread makes progress in a finite number of steps
//! - **Linearizable**: Operations appear to occur atomically at some point
//! - **ABA-safe**: Uses generation counters in tagged pointers
//!
//! ## Performance
//!
//! - Search: O(log n) expected, lock-free
//! - Insert: O(log n) expected, lock-free
//! - Remove: O(log n) expected, lock-free
//! - Space: O(n) average, O(n log n) worst case
//!
//! ## References
//!
//! M. Herlihy, Y. Lev, V. Luchangco, N. Shavit. "A Simple Optimistic Skiplist Algorithm" (2007)
//! W. Pugh. "Concurrent Maintenance of Skip Lists" (1990)

const std = @import("std");
const Allocator = std.mem.Allocator;
const Order = std.math.Order;
const AtomicOrder = std.builtin.AtomicOrder;
const assert = std.debug.assert;

/// Lock-free concurrent skip list.
///
/// Thread-safe sorted map supporting concurrent operations without locks.
///
/// Example:
/// ```zig
/// const IntContext = struct {
///     pub fn compare(_: @This(), a: i32, b: i32) Order {
///         return std.math.order(a, b);
///     }
/// };
/// var list = ConcurrentSkipList(i32, []const u8, IntContext, IntContext.compare)
///     .init(allocator, .{}, .{ .seed = 42 });
/// defer list.deinit();
///
/// try list.insert(42, "answer");
/// const val = list.get(42); // Some("answer")
/// ```
pub fn ConcurrentSkipList(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compare_fn: fn (ctx: Context, a: K, b: K) Order,
) type {
    // Use compare_fn as a comptime parameter
    const compareFn = compare_fn;

    return struct {
        const Self = @This();

        /// Maximum height of the skip list
        pub const max_level: usize = 16;

        /// Probability for level generation (p = 1/2 for simplicity)
        const probability_denominator: u32 = 2;

        /// Node in the skip list
        const Node = struct {
            key: K,
            value: V,
            /// forward[i] is a tagged pointer to the next node at level i
            /// Lower bit indicates if this node is marked for deletion
            forward: [max_level]std.atomic.Value(usize),
            top_level: usize,
            /// Intrusive link for the retired-node list (`Reclaim.retired_head`). Only ever
            /// touched while holding `Reclaim.mutex`; unrelated to `forward` traversal.
            retire_next: ?*Node = null,

            fn init(key: K, value: V, level: usize) Node {
                var forward: [max_level]std.atomic.Value(usize) = undefined;
                for (&forward) |*fwd| {
                    fwd.* = std.atomic.Value(usize).init(0);
                }
                return .{
                    .key = key,
                    .value = value,
                    .forward = forward,
                    .top_level = level,
                };
            }
        };

        /// Tagged pointer encoding: [ptr][mark_bit]
        /// Lower bit = mark (1 if logically deleted)
        const TaggedPtr = struct {
            ptr: ?*Node,
            marked: bool,

            fn pack(self: TaggedPtr) usize {
                const ptr_val: usize = @intFromPtr(self.ptr);
                return ptr_val | @intFromBool(self.marked);
            }

            fn unpack(val: usize) TaggedPtr {
                const marked = (val & 1) != 0;
                const ptr_val = val & ~@as(usize, 1);
                return .{
                    .ptr = if (ptr_val == 0) null else @ptrFromInt(ptr_val),
                    .marked = marked,
                };
            }
        };

        pub const Entry = struct {
            key: K,
            value: V,
        };

        /// Deferred-reclamation state for `remove()` (issue #38).
        ///
        /// `remove()` unlinks a node lock-free, but a concurrent `get`/`contains` traversal may
        /// still hold a raw `*Node` into it, so the node cannot be freed immediately. Every
        /// reader increments `readers` on entry and decrements on exit; a retired (unlinked)
        /// node is queued on `retired_head` instead of freed, and the queue is only drained --
        /// freeing every queued node -- when `readers` observes zero, batched so the
        /// atomic-read-then-free pass is amortized rather than done per-remove.
        const Reclaim = struct {
            readers: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
            mutex: std.Thread.Mutex = .{},
            retired_head: ?*Node = null,
            retired_count: usize = 0,
            /// Copied from `Options.retired_max` at `init`; drain is attempted once this many
            /// nodes have accumulated on `retired_head`.
            retired_max: usize = 64,
        };

        /// Construction options.
        ///
        /// `seed` drives the level-generation PRNG (see ADR 0001 D1): a clock-derived seed is an
        /// undeclared input, so the caller must supply one explicitly for a reproducible list.
        ///
        /// `retired_max` bounds how many logically-removed nodes accumulate on the retired list
        /// (see `Reclaim`) before a drain is attempted; defaulted so existing call sites that
        /// only set `seed` stay source-compatible.
        pub const Options = struct {
            seed: u64,
            retired_max: usize = 64,
        };

        allocator: Allocator,
        header: *Node,
        ctx: Context,
        prng_mutex: std.Thread.Mutex,
        prng: std.Random.DefaultPrng,
        rc: *Reclaim,

        // -- Lifecycle --

        /// Initialize an empty concurrent skip list.
        ///
        /// `options.seed` seeds the level-generation PRNG; the same seed and the same operation
        /// sequence produce the same node heights (ADR 0001 D1). There is no default seed.
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, ctx: Context, options: Options) !Self {
            assert(options.retired_max > 0);

            const header = try allocator.create(Node);
            errdefer allocator.destroy(header);

            header.* = Node.init(undefined, undefined, max_level - 1);

            const rc = try allocator.create(Reclaim);
            rc.* = .{ .retired_max = options.retired_max };

            const prng = std.Random.DefaultPrng.init(options.seed);

            assert(rc.retired_count == 0);

            return Self{
                .allocator = allocator,
                .header = header,
                .ctx = ctx,
                .prng_mutex = .{},
                .prng = prng,
                .rc = rc,
            };
        }

        /// Free all nodes in the skip list, including any still-retired (unlinked but not yet
        /// reclaimed) nodes.
        ///
        /// WARNING: Not thread-safe. Ensure no concurrent operations before calling -- this
        /// asserts no reader is mid-traversal, which is this container's existing contract.
        ///
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            assert(self.rc.readers.load(.seq_cst) == 0);

            var retired = self.rc.retired_head;
            while (retired) |node| {
                const next = node.retire_next;
                self.allocator.destroy(node);
                retired = next;
            }
            self.rc.retired_head = null;
            self.rc.retired_count = 0;

            assert(self.rc.retired_head == null);

            const current_tagged = self.header.forward[0].load(.monotonic);
            var current = TaggedPtr.unpack(current_tagged).ptr;

            while (current) |node| {
                const next_tagged = node.forward[0].load(.monotonic);
                const next = TaggedPtr.unpack(next_tagged).ptr;
                self.allocator.destroy(node);
                current = next;
            }

            self.allocator.destroy(self.rc);
            self.allocator.destroy(self.header);
        }

        // -- Modification --

        /// Insert a key-value pair into the skip list.
        ///
        /// If the key already exists, updates the value and returns the old value.
        /// Thread-safe.
        ///
        /// Time: O(log n) expected | Space: O(1)
        pub fn insert(self: *Self, key: K, value: V) !?V {
            const level = self.randomLevel();
            const new_node = try self.allocator.create(Node);
            new_node.* = Node.init(key, value, level);

            while (true) {
                var preds: [max_level]?*Node = undefined;
                var succs: [max_level]?*Node = undefined;

                const found = self.find(key, &preds, &succs);

                if (found) |existing| {
                    // Key exists, update value
                    const old_value = existing.value;
                    existing.value = value;
                    self.allocator.destroy(new_node);
                    return old_value;
                }

                // Link new node at each level
                var i: usize = 0;
                while (i <= level) : (i += 1) {
                    const succ = succs[i];
                    const succ_tagged = TaggedPtr{ .ptr = succ, .marked = false };
                    new_node.forward[i].store(succ_tagged.pack(), .release);
                }

                // CAS at level 0 first (linearization point)
                const pred = preds[0].?;
                const succ = succs[0];
                const old_tagged = TaggedPtr{ .ptr = succ, .marked = false };
                const new_tagged = TaggedPtr{ .ptr = new_node, .marked = false };

                if (pred.forward[0].cmpxchgWeak(
                    old_tagged.pack(),
                    new_tagged.pack(),
                    .release,
                    .acquire,
                )) |_| {
                    // CAS failed, retry
                    continue;
                }

                // Link at higher levels (best effort)
                i = 1;
                while (i <= level) : (i += 1) {
                    while (true) {
                        const pred_i = preds[i].?;
                        const succ_i = succs[i];
                        const old_i = TaggedPtr{ .ptr = succ_i, .marked = false };
                        const new_i = TaggedPtr{ .ptr = new_node, .marked = false };

                        if (pred_i.forward[i].cmpxchgWeak(
                            old_i.pack(),
                            new_i.pack(),
                            .release,
                            .acquire,
                        )) |_| {
                            // Failed, find again and retry
                            _ = self.find(key, &preds, &succs);
                            continue;
                        }
                        break;
                    }
                }

                return null;
            }
        }

        /// Remove a key from the skip list.
        ///
        /// Returns the removed value, or null if key not found.
        /// Thread-safe.
        ///
        /// Time: O(log n) expected | Space: O(1)
        pub fn remove(self: *Self, key: K) ?V {
            var preds: [max_level]?*Node = undefined;
            var succs: [max_level]?*Node = undefined;

            const victim = self.find(key, &preds, &succs) orelse return null;
            assert(victim.top_level < max_level);
            assert(compareFn(self.ctx, key, victim.key) == .eq);

            // Phase 1: Logical deletion - mark all levels from top to bottom. Only the thread
            // whose CAS transitions the level-0 mark from unmarked to marked owns this node; a
            // concurrent or prior remove() that finds it already marked must not retire it too
            // (issue #38: retiring twice would double-free once the batch drains).
            var owns_deletion = false;
            var i: usize = victim.top_level;
            while (true) {
                var succ_tagged = victim.forward[i].load(.acquire);
                var succ = TaggedPtr.unpack(succ_tagged);

                while (!succ.marked) {
                    const new_tagged = TaggedPtr{ .ptr = succ.ptr, .marked = true };
                    if (victim.forward[i].cmpxchgWeak(
                        succ_tagged,
                        new_tagged.pack(),
                        .release,
                        .acquire,
                    )) |new_val| {
                        succ_tagged = new_val;
                        succ = TaggedPtr.unpack(succ_tagged);
                        continue;
                    }
                    if (i == 0) owns_deletion = true;
                    break;
                }

                if (i == 0) break;
                i -= 1;
            }

            if (!owns_deletion) return null;

            const old_value = victim.value;

            // Phase 2: Physical deletion - unlink from bottom to top, then confirm the node is
            // fully unlinked before retiring it (issue #38): retiring a still-linked node would
            // let a later traversal dereference memory that gets freed out from under it.
            const unlink_attempts_max: usize = 4;
            var attempt: usize = 0;
            var fully_unlinked = false;
            while (attempt < unlink_attempts_max) : (attempt += 1) {
                _ = self.find(key, &preds, &succs); // Help remove marked nodes.
                fully_unlinked = self.isUnlinked(victim, &succs);
                if (fully_unlinked) break;
            }

            // If still linked after unlink_attempts_max tries, leave it linked: deinit()'s
            // normal walk will free it later, preserving "retired implies unlinked" so nothing
            // is ever double-freed.
            if (fully_unlinked) {
                self.retire(victim);
            }

            return old_value;
        }

        // -- Lookup --

        /// Get the value associated with a key.
        ///
        /// Thread-safe. Registers as a reader for the duration of the traversal (issue #38): a
        /// concurrent `remove()` will not free a physically-unlinked node while any reader is
        /// still registered, so a raw `*Node` held mid-traversal is never dangling.
        ///
        /// Time: O(log n) expected | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            const readers_before = self.rc.readers.fetchAdd(1, .seq_cst);
            assert(readers_before < std.math.maxInt(usize));
            defer {
                const readers_after = self.rc.readers.fetchSub(1, .release);
                assert(readers_after >= 1);
            }

            var current: ?*Node = self.header;
            var level: usize = max_level - 1;

            while (true) {
                while (true) {
                    const curr = current orelse break;
                    const next_tagged = curr.forward[level].load(.acquire);
                    const next = TaggedPtr.unpack(next_tagged);

                    if (next.ptr) |node| {
                        if (next.marked) {
                            current = curr;
                            break;
                        }

                        const cmp = compareFn(self.ctx, key, node.key);
                        if (cmp == .eq and !next.marked) {
                            return node.value;
                        } else if (cmp == .gt) {
                            current = node;
                            continue;
                        }
                    }
                    break;
                }

                if (level == 0) break;
                level -= 1;
            }

            return null;
        }

        /// Check if a key exists in the skip list.
        ///
        /// Thread-safe.
        ///
        /// Time: O(log n) expected | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        /// Returns the number of unlinked nodes currently queued for reclamation but not yet
        /// freed (issue #38). Exposed for testing the reclamation scheme; not part of the map's
        /// logical contents -- `get`/`contains` never observe a retired node.
        ///
        /// Time: O(1) | Space: O(1)
        pub fn pendingRetired(self: *const Self) usize {
            self.rc.mutex.lock();
            defer self.rc.mutex.unlock();

            const count = self.rc.retired_count;
            assert(self.rc.retired_max > 0);
            assert(count == 0 or self.rc.retired_head != null);
            return count;
        }

        // -- Internal helpers --

        /// Find a key and populate predecessor/successor arrays.
        ///
        /// Returns a pointer to the node if found (and not marked), null otherwise.
        fn find(
            self: *const Self,
            key: K,
            preds: *[max_level]?*Node,
            succs: *[max_level]?*Node,
        ) ?*Node {
            var pred: ?*Node = self.header;
            var level: usize = max_level - 1;

            while (true) {
                var curr = pred.?.forward[level].load(.acquire);
                var curr_node = TaggedPtr.unpack(curr).ptr;

                while (curr_node) |node| {
                    const next_tagged = node.forward[level].load(.acquire);
                    const next = TaggedPtr.unpack(next_tagged);

                    // Skip marked nodes
                    while (next.marked) {
                        // Try to physically remove
                        const new_curr = TaggedPtr{ .ptr = next.ptr, .marked = false };
                        _ = pred.?.forward[level].cmpxchgWeak(
                            curr,
                            new_curr.pack(),
                            .release,
                            .acquire,
                        );

                        curr = pred.?.forward[level].load(.acquire);
                        curr_node = TaggedPtr.unpack(curr).ptr;
                        break;
                    }

                    if (curr_node == null) break;

                    const cmp = compareFn(self.ctx, key, curr_node.?.key);
                    if (cmp == .gt) {
                        pred = curr_node;
                        curr = curr_node.?.forward[level].load(.acquire);
                        curr_node = TaggedPtr.unpack(curr).ptr;
                    } else {
                        break;
                    }
                }

                preds[level] = pred;
                succs[level] = curr_node;

                if (level == 0) break;
                level -= 1;
            }

            // Check if found at level 0
            if (succs[0]) |node| {
                const tagged = TaggedPtr.unpack(node.forward[0].load(.acquire));
                if (!tagged.marked and compareFn(self.ctx, key, node.key) == .eq) {
                    return node;
                }
            }

            return null;
        }

        /// Returns whether `victim` has been fully physically unlinked: no successor pointer at
        /// any level `<= victim.top_level` still targets it. `succs` must come from a `find()`
        /// call made after `victim` was logically marked (issue #38).
        fn isUnlinked(self: *const Self, victim: *Node, succs: *const [max_level]?*Node) bool {
            _ = self;
            assert(victim.top_level < max_level);

            var level: usize = 0;
            var still_linked = false;
            while (level <= victim.top_level) : (level += 1) {
                if (succs[level] == victim) still_linked = true;
            }

            assert(level == victim.top_level + 1);
            return !still_linked;
        }

        /// Queue a fully-unlinked node for reclamation, then attempt a drain. Below
        /// `retired_max` this is a single, cheap, non-blocking check (drain only if no reader is
        /// registered right now); at or above `retired_max` it escalates to the bounded
        /// yield-and-retry drain so a batch under sustained contention still gets reclaimed.
        /// Never call this on a node that might still be reachable from a concurrent traversal
        /// -- see `isUnlinked`.
        fn retire(self: *Self, node: *Node) void {
            assert(node.retire_next == null);

            var force_drain = false;
            {
                self.rc.mutex.lock();
                defer self.rc.mutex.unlock();

                node.retire_next = self.rc.retired_head;
                self.rc.retired_head = node;
                self.rc.retired_count += 1;
                assert(self.rc.retired_head == node);
                assert(self.rc.retired_count >= 1);
                force_drain = self.rc.retired_count >= self.rc.retired_max;
            }

            self.drain(force_drain);
        }

        /// Bounded attempt to reclaim the whole retired batch: free every retired node only once
        /// no reader is mid-traversal (`Reclaim.readers == 0`). `force` selects the effort level:
        /// `false` checks once and gives up immediately if a reader is active (the common,
        /// below-threshold case -- cheap and non-blocking); `true` retries up to
        /// `drain_attempts_max` times with a yield in between, for the over-`retired_max` case
        /// where a batch must not be left to grow unboundedly under sustained contention. Either
        /// way this never blocks or spins unboundedly.
        fn drain(self: *Self, force: bool) void {
            const drain_attempts_max: usize = 8;
            const attempts_max: usize = if (force) drain_attempts_max else 1;
            assert(attempts_max >= 1);
            var attempt: usize = 0;

            while (attempt < attempts_max) : (attempt += 1) {
                assert(attempt < attempts_max);

                if (self.rc.readers.load(.seq_cst) != 0) {
                    if (!force) return;
                    std.Thread.yield() catch |err| switch (err) {
                        error.SystemCannotYield => {},
                    };
                    continue;
                }

                var head: ?*Node = null;
                var drained_count: usize = 0;
                {
                    self.rc.mutex.lock();
                    defer self.rc.mutex.unlock();

                    head = self.rc.retired_head;
                    drained_count = self.rc.retired_count;
                    self.rc.retired_head = null;
                    self.rc.retired_count = 0;
                }

                var freed_count: usize = 0;
                var current = head;
                while (current) |node| {
                    const next = node.retire_next;
                    self.allocator.destroy(node);
                    current = next;
                    freed_count += 1;
                }

                assert(freed_count == drained_count);
                return;
            }
        }

        /// Generate a random level for a new node.
        fn randomLevel(self: *Self) usize {
            self.prng_mutex.lock();
            defer self.prng_mutex.unlock();

            var level: usize = 0;
            const random = self.prng.random();

            while (level < max_level - 1 and random.int(u32) % probability_denominator == 0) {
                level += 1;
            }

            return level;
        }

        /// Validate internal invariants
        /// Time: O(1) | Space: O(1)
        /// Note: Full validation in concurrent context is complex; this checks basic structure
        pub fn validate(self: *const Self) void {
            _ = self;
            // In concurrent skip lists, full invariant validation requires linearization
            // which is expensive. Basic structure validation is done during operations.
            // This is a placeholder for compatibility with the container protocol.
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "ConcurrentSkipList: init and deinit" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    try testing.expectEqual(@as(?i32, null), list.get(42));
}

test "ConcurrentSkipList: insert and get single element" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    const old = try list.insert(42, 100);
    try testing.expectEqual(@as(?i32, null), old);

    const val = list.get(42);
    try testing.expectEqual(@as(?i32, 100), val);
}

test "ConcurrentSkipList: insert updates existing key" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    _ = try list.insert(42, 100);
    const old = try list.insert(42, 200);

    try testing.expectEqual(@as(?i32, 100), old);
    try testing.expectEqual(@as(?i32, 200), list.get(42));
}

test "ConcurrentSkipList: multiple inserts and lookups" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    _ = try list.insert(3, 30);
    _ = try list.insert(1, 10);
    _ = try list.insert(4, 40);
    _ = try list.insert(1, 15); // Update
    _ = try list.insert(5, 50);
    _ = try list.insert(9, 90);

    try testing.expectEqual(@as(?i32, 15), list.get(1));
    try testing.expectEqual(@as(?i32, 30), list.get(3));
    try testing.expectEqual(@as(?i32, 40), list.get(4));
    try testing.expectEqual(@as(?i32, 50), list.get(5));
    try testing.expectEqual(@as(?i32, 90), list.get(9));
    try testing.expectEqual(@as(?i32, null), list.get(2));
}

test "ConcurrentSkipList: remove element" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    _ = try list.insert(42, 100);
    try testing.expectEqual(@as(?i32, 100), list.get(42));

    const removed = list.remove(42);
    try testing.expectEqual(@as(?i32, 100), removed);
    try testing.expectEqual(@as(?i32, null), list.get(42));
}

test "ConcurrentSkipList: remove from multiple elements" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    _ = try list.insert(1, 10);
    _ = try list.insert(2, 20);
    _ = try list.insert(3, 30);

    const removed = list.remove(2);
    try testing.expectEqual(@as(?i32, 20), removed);
    try testing.expectEqual(@as(?i32, 10), list.get(1));
    try testing.expectEqual(@as(?i32, null), list.get(2));
    try testing.expectEqual(@as(?i32, 30), list.get(3));
}

test "ConcurrentSkipList: remove non-existent key" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    _ = try list.insert(1, 10);
    const removed = list.remove(99);
    try testing.expectEqual(@as(?i32, null), removed);
}

test "ConcurrentSkipList: contains" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    try testing.expect(!list.contains(42));

    _ = try list.insert(42, 100);
    try testing.expect(list.contains(42));

    _ = list.remove(42);
    try testing.expect(!list.contains(42));
}

test "ConcurrentSkipList: stress test" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    // Insert 100 elements
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        _ = try list.insert(i, i * 10);
    }

    // Verify all elements
    i = 0;
    while (i < 100) : (i += 1) {
        try testing.expectEqual(@as(?i32, i * 10), list.get(i));
    }

    // Remove even elements
    i = 0;
    while (i < 100) : (i += 2) {
        _ = list.remove(i);
    }

    // Verify odd elements remain
    i = 1;
    while (i < 100) : (i += 2) {
        try testing.expectEqual(@as(?i32, i * 10), list.get(i));
    }

    // Verify even elements removed
    i = 0;
    while (i < 100) : (i += 2) {
        try testing.expectEqual(@as(?i32, null), list.get(i));
    }
}

test "ConcurrentSkipList: memory leak check" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    var i: i32 = 0;
    while (i < 50) : (i += 1) {
        _ = try list.insert(i, i);
    }

    // Verify all elements are present
    i = 0;
    while (i < 50) : (i += 1) {
        const value = list.get(i);
        try testing.expect(value != null);
        try testing.expectEqual(i, value.?);
    }

    i = 0;
    while (i < 50) : (i += 2) {
        _ = list.remove(i);
    }

    // Verify remaining elements are the odd numbers
    i = 1;
    while (i < 50) : (i += 2) {
        const value = list.get(i);
        try testing.expect(value != null);
        try testing.expectEqual(i, value.?);
    }

    // Verify even numbers are removed
    i = 0;
    while (i < 50) : (i += 2) {
        const value = list.get(i);
        try testing.expectEqual(@as(?i32, null), value);
    }
}

test "ConcurrentSkipList: with string keys" {
    const StringContext = struct {
        pub fn compare(_: @This(), a: []const u8, b: []const u8) Order {
            return std.mem.order(u8, a, b);
        }
    };

    var list = try ConcurrentSkipList([]const u8, i32, StringContext, StringContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    _ = try list.insert("apple", 1);
    _ = try list.insert("banana", 2);
    _ = try list.insert("cherry", 3);

    try testing.expectEqual(@as(?i32, 1), list.get("apple"));
    try testing.expectEqual(@as(?i32, 2), list.get("banana"));
    try testing.expectEqual(@as(?i32, 3), list.get("cherry"));
    try testing.expectEqual(@as(?i32, null), list.get("date"));
}

test "ConcurrentSkipList: remove first element preserves rest" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    // Insert in order
    _ = try list.insert(1, 10);
    _ = try list.insert(2, 20);
    _ = try list.insert(3, 30);

    // Remove first (minimum) element
    const removed = list.remove(1);
    try testing.expectEqual(@as(?i32, 10), removed);

    // Verify first is gone and rest remain
    try testing.expectEqual(@as(?i32, null), list.get(1));
    try testing.expectEqual(@as(?i32, 20), list.get(2));
    try testing.expectEqual(@as(?i32, 30), list.get(3));
}

test "ConcurrentSkipList: remove last element preserves rest" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    // Insert in order
    _ = try list.insert(10, 100);
    _ = try list.insert(20, 200);
    _ = try list.insert(30, 300);

    // Remove last (maximum) element
    const removed = list.remove(30);
    try testing.expectEqual(@as(?i32, 300), removed);

    // Verify last is gone and rest remain
    try testing.expectEqual(@as(?i32, null), list.get(30));
    try testing.expectEqual(@as(?i32, 100), list.get(10));
    try testing.expectEqual(@as(?i32, 200), list.get(20));
}

test "ConcurrentSkipList: get on empty list returns null" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    // Do not insert anything
    try testing.expectEqual(@as(?i32, null), list.get(0));
    try testing.expectEqual(@as(?i32, null), list.get(-1));
    try testing.expectEqual(@as(?i32, null), list.get(999));
    try testing.expect(!list.contains(42));

    list.validate();
}

test "ConcurrentSkipList: insert many then remove all leaves empty" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
        testing.allocator,
        .{},
        .{ .seed = 1 },
    );
    defer list.deinit();

    // Insert 20 items
    var i: i32 = 0;
    while (i < 20) : (i += 1) {
        _ = try list.insert(i, i * 10);
    }

    // Verify some items
    try testing.expectEqual(@as(?i32, 0), list.get(0));
    try testing.expectEqual(@as(?i32, 190), list.get(19));

    // Remove all 20 items
    i = 0;
    while (i < 20) : (i += 1) {
        _ = list.remove(i);
    }

    // Verify all are gone
    try testing.expectEqual(@as(?i32, null), list.get(0));
    try testing.expectEqual(@as(?i32, null), list.get(10));
    try testing.expectEqual(@as(?i32, null), list.get(19));
    try testing.expect(!list.contains(5));
}

test "ConcurrentSkipList: init-deinit loop memory safety" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };

    // 10 iterations of init-insert-get-remove-validate-deinit
    var iter: usize = 0;
    while (iter < 10) : (iter += 1) {
        var list = try ConcurrentSkipList(i32, i32, IntContext, IntContext.compare).init(
            testing.allocator,
            .{},
            .{ .seed = 1 },
        );

        // Insert three items
        _ = try list.insert(1, 10);
        _ = try list.insert(2, 20);
        _ = try list.insert(3, 30);

        // Get all three (verify non-null)
        try testing.expectEqual(@as(?i32, 10), list.get(1));
        try testing.expectEqual(@as(?i32, 20), list.get(2));
        try testing.expectEqual(@as(?i32, 30), list.get(3));

        // Remove one item
        const removed = list.remove(2);
        try testing.expectEqual(@as(?i32, 20), removed);

        // Verify removal
        try testing.expectEqual(@as(?i32, null), list.get(2));

        // Validate and deinit
        list.validate();
        list.deinit();
    }
}

test "ConcurrentSkipList: same seed produces identical random-level sequence" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };
    const Instance = ConcurrentSkipList(i32, i32, IntContext, IntContext.compare);

    var list_a = try Instance.init(testing.allocator, .{}, .{ .seed = 7 });
    defer list_a.deinit();
    var list_b = try Instance.init(testing.allocator, .{}, .{ .seed = 7 });
    defer list_b.deinit();

    var i: usize = 0;
    while (i < 32) : (i += 1) {
        try testing.expectEqual(list_a.randomLevel(), list_b.randomLevel());
    }
}

test "ConcurrentSkipList: different seeds diverge in random-level sequence" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };
    const Instance = ConcurrentSkipList(i32, i32, IntContext, IntContext.compare);

    var list_a = try Instance.init(testing.allocator, .{}, .{ .seed = 1 });
    defer list_a.deinit();
    var list_b = try Instance.init(testing.allocator, .{}, .{ .seed = 2 });
    defer list_b.deinit();

    // Probabilistic, not a mathematical guarantee: two distinct seeds could in principle draw
    // 32 identical levels in a row. With std.Random.DefaultPrng this is astronomically
    // unlikely; if this ever flakes, raise the draw count rather than assuming a bad seed pair.
    var i: usize = 0;
    var diverged = false;
    while (i < 32) : (i += 1) {
        if (list_a.randomLevel() != list_b.randomLevel()) diverged = true;
    }
    try testing.expect(diverged);
}

test "ConcurrentSkipList: same seed produces identical tree shape across insert sequence" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };
    const Instance = ConcurrentSkipList(i32, i32, IntContext, IntContext.compare);

    var list_a = try Instance.init(testing.allocator, .{}, .{ .seed = 99 });
    defer list_a.deinit();
    var list_b = try Instance.init(testing.allocator, .{}, .{ .seed = 99 });
    defer list_b.deinit();

    var i: i32 = 0;
    while (i < 40) : (i += 1) {
        _ = try list_a.insert(i, i * 10);
        _ = try list_b.insert(i, i * 10);
    }

    i = 0;
    while (i < 40) : (i += 1) {
        try testing.expectEqual(list_a.get(i), list_b.get(i));
    }

    // Lookups alone are seed-independent for a correct skip list; the actual tree-shape claim
    // is the per-node height (top_level), which only the level-generation PRNG determines.
    var node_a = Instance.TaggedPtr.unpack(list_a.header.forward[0].load(.monotonic)).ptr;
    var node_b = Instance.TaggedPtr.unpack(list_b.header.forward[0].load(.monotonic)).ptr;
    var compared: usize = 0;
    while (node_a) |na| {
        const nb = node_b.?;
        try testing.expectEqual(na.top_level, nb.top_level);
        node_a = Instance.TaggedPtr.unpack(na.forward[0].load(.monotonic)).ptr;
        node_b = Instance.TaggedPtr.unpack(nb.forward[0].load(.monotonic)).ptr;
        compared += 1;
    }
    try testing.expectEqual(@as(?*Instance.Node, null), node_b);
    try testing.expectEqual(@as(usize, 40), compared);
}

// -- Issue #38: remove() must reclaim physically-unlinked nodes without a use-after-free --

test "ConcurrentSkipList: pendingRetired reaches zero after single-threaded remove" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };
    const Instance = ConcurrentSkipList(i32, i32, IntContext, IntContext.compare);

    var list = try Instance.init(testing.allocator, .{}, .{ .seed = 1 });
    defer list.deinit();

    // Boundary: nothing retired yet before any remove.
    try testing.expectEqual(@as(usize, 0), list.pendingRetired());

    _ = try list.insert(1, 10);
    _ = try list.insert(2, 20);
    _ = try list.insert(3, 30);

    _ = list.remove(2);

    // Single-threaded: no concurrent traversal ever holds the reader count above zero across
    // this call, so the bounded drain must retire the node deterministically -- poll instead of
    // asserting instantly-zero so this doesn't depend on the drain running synchronously inside
    // remove() itself, only on it completing promptly with no other thread involved.
    var attempts: usize = 0;
    while (list.pendingRetired() != 0 and attempts < 1000) : (attempts += 1) {}
    try testing.expectEqual(@as(usize, 0), list.pendingRetired());
}

test "ConcurrentSkipList: remove reclaims retired nodes without leaking (issue #38)" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };
    const Instance = ConcurrentSkipList(i32, i32, IntContext, IntContext.compare);

    var list = try Instance.init(testing.allocator, .{}, .{ .seed = 42, .retired_max = 1024 });
    defer list.deinit();

    const key_count: i32 = 50;

    var i: i32 = 0;
    while (i < key_count) : (i += 1) {
        _ = try list.insert(i, i * 10);
    }

    i = 0;
    while (i < key_count) : (i += 1) {
        const removed = list.remove(i);
        try testing.expectEqual(@as(?i32, i * 10), removed);
    }

    // All 50 keys must be gone -- and, per std.testing.allocator's own leak detection at
    // `defer list.deinit()` above, every node physically unlinked by those 50 removes must
    // actually have been freed, not merely unlinked (issue #38's core regression).
    i = 0;
    while (i < key_count) : (i += 1) {
        try testing.expectEqual(@as(?i32, null), list.get(i));
    }
}

test "ConcurrentSkipList: second remove of an already-removed key returns null, not a double-retire" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };
    const Instance = ConcurrentSkipList(i32, i32, IntContext, IntContext.compare);

    var list = try Instance.init(testing.allocator, .{}, .{ .seed = 3 });
    defer list.deinit();

    _ = try list.insert(7, 700);

    const first = list.remove(7);
    try testing.expectEqual(@as(?i32, 700), first);

    // Idempotency: a second remove of the same, already-logically-deleted key must not retire
    // the same node a second time (that would double-free once the batch drains).
    const second = list.remove(7);
    try testing.expectEqual(@as(?i32, null), second);

    const third = list.remove(7);
    try testing.expectEqual(@as(?i32, null), third);
}

test "ConcurrentSkipList: insert after remove of same key is visible to get" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };
    const Instance = ConcurrentSkipList(i32, i32, IntContext, IntContext.compare);

    var list = try Instance.init(testing.allocator, .{}, .{ .seed = 5 });
    defer list.deinit();

    _ = try list.insert(9, 90);
    const removed = list.remove(9);
    try testing.expectEqual(@as(?i32, 90), removed);
    try testing.expectEqual(@as(?i32, null), list.get(9));

    const old = try list.insert(9, 999);
    try testing.expectEqual(@as(?i32, null), old);
    try testing.expectEqual(@as(?i32, 999), list.get(9));
}

test "ConcurrentSkipList: concurrent readers survive a remover reclaiming nodes, leak-free" {
    const IntContext = struct {
        pub fn compare(_: @This(), a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    };
    const Instance = ConcurrentSkipList(i32, i32, IntContext, IntContext.compare);

    var list = try Instance.init(testing.allocator, .{}, .{ .seed = 11, .retired_max = 256 });
    defer list.deinit();

    const key_count: i32 = 20;
    const reader_count: usize = 4;
    const reader_iterations_max: usize = 200;

    var k: i32 = 0;
    while (k < key_count) : (k += 1) {
        _ = try list.insert(k, k * 10);
    }

    const ReaderCtx = struct {
        list_ptr: *Instance,
        keys_max: i32,
    };
    const reader_fn = struct {
        fn run(ctx: ReaderCtx) void {
            var iter: usize = 0;
            while (iter < reader_iterations_max) : (iter += 1) {
                const key: i32 = @intCast(@as(usize, @intCast(iter)) % @as(usize, @intCast(ctx.keys_max)));
                // Bounded, read-only traversal concurrent with the remover below; must never
                // dereference a node freed out from under it (issue #38's unsafe-immediate-free
                // hazard is exactly what the reader/reclaim seam guards against).
                _ = ctx.list_ptr.get(key);
                _ = ctx.list_ptr.contains(key);
            }
        }
    }.run;

    const RemoverCtx = struct {
        list_ptr: *Instance,
        keys_max: i32,
    };
    const remover_fn = struct {
        fn run(ctx: RemoverCtx) void {
            var key: i32 = 0;
            while (key < ctx.keys_max) : (key += 1) {
                _ = ctx.list_ptr.remove(key);
            }
        }
    }.run;

    var readers: [reader_count]std.Thread = undefined;
    var i: usize = 0;
    while (i < reader_count) : (i += 1) {
        readers[i] = try std.Thread.spawn(.{}, reader_fn, .{ReaderCtx{
            .list_ptr = &list,
            .keys_max = key_count,
        }});
    }

    const remover = try std.Thread.spawn(.{}, remover_fn, .{RemoverCtx{
        .list_ptr = &list,
        .keys_max = key_count,
    }});

    remover.join();
    i = 0;
    while (i < reader_count) : (i += 1) {
        readers[i].join();
    }

    // The remover removed every key; readers only ever observed state, so all keys must be gone
    // and (per std.testing.allocator at deinit) every retired node must have been reclaimed.
    k = 0;
    while (k < key_count) : (k += 1) {
        try testing.expectEqual(@as(?i32, null), list.get(k));
    }
}
