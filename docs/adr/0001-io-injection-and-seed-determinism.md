# 0001 — `io: Io` injection, and seeds are not I/O

- **Status**: accepted
- **Date**: 2026-09-10
- **Plan item**: `docs/plans/001-zig-0.16-and-tiger-baseline.md`, "Fix the public-API shape once"
- **Version impact**: MAJOR — this is the v3.0.0 break enumeration
- **Rule**: `citadel/core/rules/zig-0.16.md` § THE KINGDOM CONVENTION for `io: Io`

## Context

Zig 0.16 deletes `std.time`'s clocks, most of `std.fs`, and every `std.Thread` sync primitive
from their namespaces and reissues them through the `std.Io` vtable (`Io.Clock`, `Io.Dir` /
`Io.File`, `Io.Mutex` / `Io.Group`). The kingdom convention says `io: Io` is the first parameter
after the receiver on every public function that touches filesystem, network, time, sleep, sync
or process APIs; leaf functions take it per call; libraries never construct an `Io`.

zuda is 445 files / ~461k LOC with ~60 containers whose API shape (allocator-first, Managed
stores the allocator, Unmanaged takes it per call) predates the kingdom rules. Threading `io`
through that surface *is* the v3.0.0 break; getting it wrong costs a v4. The actual exposure,
measured (`grep` on `main`, 2026-09-10):

| Class | Sites | Files |
|---|---|---|
| `std.time.*` | 21 | 13 (4 containers, 5 ML, 3 harness, 1 test-only) |
| `fs.cwd()` in library code | 4 | 1 (`ndarray.zig`: `save`/`load`/`toCSV`/`fromCSV`) |
| `fs.cwd()` in tests | 29 | 1 (`ndarray.zig` round-trip tests) |
| `std.Thread.Mutex` | 2 | 2 (`concurrent_skip_list.zig`, `work_stealing_deque.zig`) |
| `std.Thread.spawn` | 1 | 1 (`work_stealing_deque.zig`, in a test) |
| `std.posix.getrandom` | 2 | 1 (`bogosort.zig`) |
| `std.io.*` | 7 | 4 (`perf.zig`, `bench.zig`, `mlp.zig`, `lightgbm.zig`) |

The decisive observation: **15 of the 21 `std.time.*` sites are PRNG seeding**, not timekeeping —
`Random.DefaultPrng.init(@intCast(std.time.timestamp()))`. One more (`bloom_filter.zig:416`) is a
throughput assertion inside a unit test. `src/algorithms/parallel/*` contains **no** `std.Thread`
at all: `parallelMergeSort`, `parallelScan`, `mapReduce` and friends take a `num_threads: usize`
they ignore (`parallel_sort.zig:29`: "Real implementation would use std.Thread.spawn"). So the
naive reading of the plan — "thread `io` wherever `std.time`/`std.Thread` appears" — would put
an `Io` parameter on roughly forty container entry points that do no I/O whatsoever, and none on
the functions that actually promise concurrency.

## Options considered

| # | Option | Blast radius | Determinism | Verdict |
|---|---|---|---|---|
| 1 | `io` on every fn that today names `std.time`/`std.Thread` | ~40 public fns, 12 files | unchanged (still clock-seeded) | rejected: pays the break, buys nothing |
| 2 | Store `io` in every container next to the allocator (extend the Managed convention) | every `init` + `deinit` | unchanged | rejected — see D2 |
| 3 | Package-level default `Io` (lazy `Io.Threaded`) | zero | worse | rejected: forbidden by the rule, and library code must not choose a runtime |
| 4 | Split the axes: `seed: u64` for non-determinism, `io: Io` only for real I/O | 4 containers, 1 ndarray file, 10 parallel fns, 3 harness files | **fixed** — every container becomes reproducible | **chosen** |

## Decision

### D1 — Seeding is not I/O. Clock-derived seeds become explicit `seed: u64` options

A PRNG seed read from the wall clock is not an I/O dependency, it is an *undeclared input*.
Tiger Style rule 7 ("inject clock, PRNG, allocator, `Io`") wants that input named. Every
`Random.DefaultPrng.init(std.time.*)` site becomes a caller-supplied seed carried in the
container's options struct; nothing in that path gains an `io` parameter. This removes 15 of 21
`std.time.*` sites, makes `SkipList`, `ConcurrentSkipList`, `RobinHoodHashMap`, `CuckooHashMap`
and the five ML estimators reproducible under seeded model-based tests, and clears the `tidy`
ban-list entries for `std.time.*` without importing `Io` into the container layer.

There is no default seed. `init(gpa, ctx)` keeping a hidden clock seed would preserve exactly the
non-determinism this ADR exists to remove, so the seed is a required field of a required options
struct — one construction path, per Tiger Style rule 4. `initWithSeed` (present today on
`SkipList`) disappears into it.

### D2 — `io` is never stored in a container; the Managed allocator convention does not extend

`REALM.md`'s Container API shape rule lets a Managed variant store its allocator. That rule does
not generalize to `io`, for four reasons:

1. **Ownership vs. execution context.** The allocator is stored because the container *owns*
   memory allocated from it and must return it in `deinit`; the identity of the allocator is part
   of the container's invariants. `io` owns nothing on the container's behalf — it is the context
   the *current call* executes in, and the same container may legitimately be driven from a test
   `Io` in one call and a threaded `Io` in the next.
2. **Lifetime.** A stored `Io` outlives the call that supplied it, which is precisely what the
   convention's "do not stash it just to save a parameter" forbids.
3. **`deinit` stays I/O-free.** zuda containers own memory, never file handles or OS resources.
   Storing `io` would make `deinit` I/O-capable by construction and invite a future `deinit(io)`.
4. **Relocatability.** Containers are `memcpy`-relocatable value types today; `NDArray` and the
   compat shims for silica/zoltraak/zr depend on that.

Therefore: **`init`/`initManaged` never take `io`.** Only the specific methods that touch time,
fs or sync take it, first after the receiver. A TTL cache, were zuda to grow one, would be
`init(gpa, options)` + `isExpired(self, io)` + `evictExpired(self, io)` — not `init(gpa, io)`.

**One exception, matching the rule's `http.Client` carve-out**: `internal/bench.zig`'s benchmark
`Runner` is a long-lived owning service object whose entire purpose is timing; it caches an `io`
field set once at construction. It is `internal/`, not public API.

### D3 — Short critical sections use `lockUncancelable`; `error.Canceled` stays out of
container error sets

`Io.Mutex.lock(io)` returns `Cancelable!void`; `Io.Mutex.lockUncancelable(io)` returns `void`
(`std/Io.zig:1602,1623`). zuda's two mutexes guard a PRNG draw and a deque resize — bounded,
allocation-free, non-blocking critical sections where cancelation has no meaning. Using
`lockUncancelable` keeps `ConcurrentSkipList.insert` at `!void` instead of promoting it to a set
that carries `error.Canceled`, which is the call-site-dimensionality rule (`void` > `bool` >
`u64` > `?u64` > `!u64`) applied to error sets. Only functions that genuinely await —
`Io.Group.await` in the parallel algorithms, and every `Io.File` operation — surface
`error.Canceled`, and they propagate it; never `else => unreachable`.

### D4 — `Io.Dir` is a parameter, not `Io.Dir.cwd()` called inside the library

`ndarray`'s four file entry points take `dir: Io.Dir` alongside `io`. `cwd()` is ambient process
state in the same way a global allocator is; a library that reaches for it makes its callers'
tests order-dependent and its own tests dependent on the process working directory. Callers write
`Io.Dir.cwd()` explicitly at the call site; tests pass `tmp.dir` and stop littering `/tmp` (29
`fs.cwd().deleteFile("/tmp/...")` calls disappear from the test block).

### D5 — The `.npy`-style binary format gains a checksum and becomes version 2

Today's format is magic `"NDAR"` + `u32` version 1 + header + raw payload — magic and version but
no integrity check, and `load` silently accepts a truncated file (`file.read` short reads are not
checked). v3.0.0 writes version 2 = the version-1 layout plus a trailing little-endian CRC-32 of
everything preceding it, and `load` verifies it. Version 1 files remain readable (no checksum
verification, documented); version 2 is what `save` emits. New error: `error.ChecksumMismatch`,
plus `error.UnexpectedEndOfFile` for the short-read case that is currently silent corruption.

### D6 — The `parallel/*` family gets `io` and `Io.Group`, and its limit becomes honest

`parallelMergeSort`, `parallelQuickSort`, `parallelPrefixSum`, `parallelReduce`, `parallelMap`,
`parallelFilter`, `parallelScan`, `mapReduce`, `groupBy`, `partition` advertise concurrency and
deliver none. Two defensible endings: delete the `num_threads` parameter and rename them to what
they are, or implement them on `Io.Group`. We implement them, because adding `io` to these
signatures later is a second major break, and because the rest of this ADR removes `io` from the
places it does not belong precisely so it can be paid where it does. `num_threads: usize`
(unbounded, ignored) becomes `options: struct { concurrency_max: usize }` — the limit is part of
the signature. These are divide-and-conquer tasks with no inter-task dependency, so `Group.async`
is correct and cannot deadlock when the `Io` implementation runs tasks inline; `io.concurrent` is
not needed and `error.ConcurrencyUnavailable` never appears.

### D7 — Signature shape, in one line each

```zig
// Non-I/O, previously clock-seeded: gains a seed, never an Io.
pub fn init(gpa: Allocator, ctx: Context, options: Options) Allocator.Error!Self

// Touches time / fs / sync: io first after the receiver, per call, never stored.
pub fn method(self: *Self, io: Io, ...) Error!T

// Free function touching I/O: io first.
pub fn parallelMap(comptime T: type, io: Io, gpa: Allocator, ...) Error![]T
```

## The v3.0.0 break, enumerated

### Group A — clock-seeded containers: **no `io`**, new required `seed` (D1)

| File | Before | After |
|---|---|---|
| `containers/lists/skip_list.zig` | `init(allocator, ctx)` / `initWithSeed(allocator, ctx, seed)` | `init(gpa, ctx, .{ .seed = s })`; `initWithSeed` removed |
| `containers/lists/concurrent_skip_list.zig` | `init(allocator, ctx)` | `init(gpa, ctx, .{ .seed = s })` |
| `containers/hashing/robin_hood_hash_map.zig` | `init(allocator, context)`, `initCapacity(allocator, context, cap)` | `init(gpa, ctx, .{ .seed = s })`, `initCapacity(gpa, ctx, .{ .seed = s, .capacity = n })` |
| `containers/hashing/cuckoo_hash_map.zig` | `init`/`initCapacity` as above; private `rehash` re-seeds from `nanoTimestamp` | same options shape; `rehash` advances a stored `reseed_prng` (seeded once at `init`) instead of reading the clock |
| `algorithms/machine_learning/{kmeans,gmm,dqn,ddpg,tsne}.zig` | `seed: ?u64 = null` falling back to the clock | `seed: u64`, required |
| `algorithms/sorting/bogosort.zig` | `bogoSort(T, arr, cmp)`, `bogoSortBounded(T, arr, cmp, max_iterations)` — `posix.getrandom` inside | `bogoSort(T, arr, cmp, seed)`, `bogoSortBounded(T, arr, cmp, seed, iterations_max)`; `std.posix` gone |
| `internal/testing.zig` | clock-seeded property-test PRNG | `std.testing.random_seed` |

### Group B — `Thread` sync: **`io` on the locking methods only** (D2, D3)

| File | Before | After |
|---|---|---|
| `containers/lists/concurrent_skip_list.zig` | `prng_mutex: std.Thread.Mutex` | `prng_mutex: Io.Mutex`; `insert(self, gpa, k, v)` → `insert(self, io, gpa, k, v)`; `remove(self, k)` → `remove(self, io, k)`. `get`/`contains`/iteration unchanged (lock-free reads) |
| `containers/queues/work_stealing_deque.zig` | `mutex: std.Thread.Mutex` | `mutex: Io.Mutex`; `push(self, item)` → `push(self, io, item)` (resize path locks); `pop`/`steal` unchanged (atomics only). Test `std.Thread.spawn` → `Io.Group.async` |

`init`/`deinit` on both stay `io`-free. `Io.Mutex` is `extern struct` and zero-initializes, so the
struct literal `.mutex = .{}` survives unchanged.

### Group C — `ndarray` filesystem: `io` + explicit `dir` (D4, D5)

| Before | After |
|---|---|
| `save(self: *const Self, path: []const u8) !void` | `save(self: *const Self, io: Io, dir: Io.Dir, path: []const u8) SaveError!void` |
| `load(allocator: mem.Allocator, path: []const u8) !Self` | `load(gpa: Allocator, io: Io, dir: Io.Dir, path: []const u8) LoadError!Self` |
| `toCSV(self: *const Self, path: []const u8, delimiter: u8) !void` | `toCSV(self: *const Self, io: Io, dir: Io.Dir, path: []const u8, options: CsvOptions) SaveError!void` |
| `fromCSV(allocator, path, delimiter) !Self` | `fromCSV(gpa: Allocator, io: Io, dir: Io.Dir, path: []const u8, options: CsvOptions) LoadError!Self` |

`CsvOptions = struct { delimiter: u8 = ',', bytes_max: usize = 100 << 20 }` — the 100 MB cap is
hardcoded inside `fromCSV` today; limits belong in the signature. Error sets become explicit and
named (both include `error.Canceled`; `LoadError` adds `ChecksumMismatch`,
`UnexpectedEndOfFile`). Internals: `fs.cwd().createFile(p, .{})` → `dir.createFile(io, p, .{})`,
`file.close()` → `file.close(io)`, `file.write(b)` → `file.writeStreamingAll(io, b)`,
`file.read(&b)` → `file.reader(io, &buf).interface.readSliceAll(&b)` (checked short reads),
`file.readToEndAlloc(gpa, n)` →
`reader.interface.allocRemaining(gpa, .limited(options.bytes_max))`. The module alias
`const io = stdlib.io;` at `ndarray.zig:39` must be **deleted** before any of this — it shadows
the `io` parameter name, and `std.io` no longer exists in 0.16.

### Group D — parallel algorithms: `io` first, `Io.Group` inside (D6)

| Before | After |
|---|---|
| `parallelMergeSort(T, allocator, arr, num_threads)` | `parallelMergeSort(T, io, gpa, arr, .{ .concurrency_max = n })` |
| `parallelQuickSort(T, arr, max_depth)` | `parallelQuickSort(T, io, arr, .{ .depth_max = d })` |
| `parallelPrefixSum(T, allocator, arr)` | `parallelPrefixSum(T, io, gpa, arr, options)` |
| `parallelReduce` / `parallelMap` / `parallelFilter` | `io` first after the comptime params, `options` last |
| `prefix_sum.parallelScan(T, input, allocator, num_threads)` | `parallelScan(T, io, gpa, input, .{ .concurrency_max = n })` |
| `map_reduce.{mapReduce,groupBy,partition}` | `io` first after the comptime params, `options` last |
| `inclusiveScan` / `exclusiveScan` / `scanInPlace` / `segmentedScan` / `reduce` | unchanged — sequential by contract, no `io` |

All four gain `error.Canceled` in their error sets (from `Group.await`), propagated, never
swallowed.

### Group E — harness (`internal/`, `utils/perf.zig`)

| Before | After |
|---|---|
| `perf.timeFn(allocator, func, args)` (allocator unused) | `perf.timeFn(io, func, args) u64` — unused `allocator` dropped |
| `perf.timeFnIters(allocator, func, args, warmup, iterations)` | `perf.timeFnIters(io, func, args, .{ .warmup = w, .iterations = n })` |
| `perf.expectFaster(...)` | `io` first |
| `perf.throughput` / `mbPerSec` / `AllocTracker` | unchanged — pure arithmetic / allocator-only |
| `perf.report(self, writer: anytype)` , `std.io.fixedBufferStream` | `report(self, w: *Io.Writer)`; fixed buffer via `Io.Writer.fixed` |
| `internal/bench.zig` `Timer`, `std.io.getStdOut()`, `std.io.AnyWriter` | `Runner` caches `io` (D2 exception); `Io.Clock.awake.now(io)` + `Timestamp.untilNow(io, .awake)`; `File.stdout().writer(io, &buf)` + explicit `flush()`; `*Io.Writer` |
| `mlp.zig` / `lightgbm.zig` `log_writer: ?std.io.AnyWriter` | `log_writer: ?*Io.Writer` |

## Spike: `src/containers/probabilistic/bloom_filter.zig`

The plan names `bloom_filter.zig` as the proof. Read in full (586 lines): it has exactly one
`std.time.*` site, `line 416`, **inside a test**. Its ten public functions —

```zig
pub fn init(allocator: std.mem.Allocator, m: usize, k: usize, ctx: Context) !Self
pub fn initWithFalsePositiveRate(allocator: std.mem.Allocator, n: usize, p: f64, ctx: Context) !Self
pub fn deinit(self: *Self) void
pub fn add(self: *Self, item: T) void
pub fn contains(self: *const Self, item: T) bool
pub fn clear(self: *Self) void
pub fn estimatedFalsePositiveRate(self: *const Self) f64
pub fn approximateCount(self: *const Self) usize
pub fn unionWith(self: *Self, other: *const Self) error{IncompatibleFilters}!void
pub fn intersectionWith(self: *Self, other: *const Self) error{IncompatibleFilters}!void
pub fn validate(self: *const Self) !void
```

— touch no clock, no filesystem, no sync primitive, and therefore **take no `io` and are
byte-for-byte unchanged in v3.0.0**. `BloomFilter` is a bit array with comptime hash functions;
it is pure.

That is the point of the spike. Under option 1 above, `bloom_filter.zig` would have been dragged
into the break because a *test* used a timer. Its correct v3.0.0 diff is:

```zig
// before (test "BloomFilter - benchmark calculation verification", lines 416-431)
var timer = try std.time.Timer.start();
for (0..100_000) |i| _ = filter.contains(@intCast(i % 2000));
const elapsed_ns = timer.read();
try testing.expect(elapsed_ns > 0);
const ops_per_sec = @divFloor(100_000 * 1_000_000_000, elapsed_ns);
try testing.expect(@divFloor(ops_per_sec, 1_000_000) >= 1);   // wall-clock throughput assertion

// after — no timer, no io, no wall clock in a unit test
const lookups: u64 = 100_000;
var found: usize = 0;
for (0..lookups) |i| found += @intFromBool(filter.contains(@intCast(i % 2000)));
try testing.expect(found >= 1000);                            // 1000 members are present
try testing.expectEqual(@as(u64, 100_000), perf.throughput(lookups, 1_000_000_000));
```

The throughput *arithmetic* is what the test was really guarding (it was written to verify a
benchmark fix); it is a pure function and is tested as one. The wall-clock measurement moves to
`bench/`, where an injected `Io` belongs and where a machine-speed assertion is not flaky CI.
Result: `bloom_filter.zig` compiles on 0.16 with zero `Io` references, and
`grep 'std\.time\.' src/containers/probabilistic/bloom_filter.zig` is empty.

**`bloom_filter.zig` alone is an insufficient spike** — it proves the negative space only. The
positive-space spike must be `containers/lists/concurrent_skip_list.zig`: it is the single file
exercising every clause of this ADR at once (D1 seed, D2 no stored `io`, D3
`lockUncancelable`) and it carries the largest signature blast radius of any container. Both
files should land in the same PR as the shape proof.

## Migration and parity tests

1. **Seed determinism (new, Group A).** For each of `SkipList`, `ConcurrentSkipList`,
   `RobinHoodHashMap`, `CuckooHashMap`, `KMeans`, `GMM`, `TSNE`: build two instances with the
   same seed, apply the same operation sequence, assert identical iteration order / identical
   cluster assignment. Second half: two different seeds produce a different internal layout for
   the same key set (negative space — proves the seed is actually reaching the PRNG).
2. **Cuckoo rehash determinism.** Force ≥ 3 rehashes with a fixed seed and assert the resulting
   `seed1`/`seed2` pair is reproducible across runs — the current `nanoTimestamp` re-seed is not.
3. **Format parity (D5).** Golden version-1 file committed under `tests/fixtures/`: `load` reads
   it, round-trips through `save`, and the reloaded array equals the original. Corrupt one payload
   byte of a version-2 file → `error.ChecksumMismatch`. Truncate a version-2 file →
   `error.UnexpectedEndOfFile`. Both must fail *before* this ADR's `load` is trusted.
4. **`Io.Dir` parity (D4).** Every `ndarray` fs test moves from `fs.cwd()` + `/tmp` literals to
   `std.testing.tmpDir(.{})` (options type is `Io.Dir.OpenOptions` in 0.16) and
   `std.testing.io`. Assert the 29 `deleteFile("/tmp/...")` calls are gone.
5. **Cancelation (D3, D6).** With a cancelling `Io`, assert `parallelMergeSort` returns
   `error.Canceled` and leaks nothing (`testing.allocator` catches it), while
   `ConcurrentSkipList.insert` — built on `lockUncancelable` — cannot return it, enforced by its
   declared error set at compile time.
6. **Concurrency parity (D6).** `parallelMergeSort`/`parallelScan`/`mapReduce` results must equal
   their sequential counterparts for the same input across `concurrency_max` ∈ {1, 2, 8}, on a
   seeded random input corpus.
7. **Cross-compile.** Group B and D verified on macOS **and** an `x86_64-linux` cross-compile;
   `LockFreeStack`/`LockFreeQueue` remain macOS-only by design (`memory/debugging.md`).

## Consequences

**Positive.**
- Every container that used to seed itself from the wall clock becomes reproducible, which is the
  precondition for the model-based seeded tests Tiger Style rule 7 asks for.
- `io` appears on ~18 public functions instead of ~40, and in 4 files' internals instead of 12.
  60-odd containers (including all of `trees/`, `heaps/`, `spatial/`, `strings/`, `graphs/`,
  `probabilistic/`) have an **unchanged** public API; consumers recompile rather than edit.
- 20 of 21 `std.time.*` sites and both `std.posix` sites leave `src/`, retiring three `tidy`
  ban-list classes and moving the "gate `test` on `tidy`" plan item measurably closer.
- The `ndarray` binary format finally satisfies magic + version + checksum, and short reads stop
  being silent corruption.
- `parallel/*` stops lying about concurrency.

**Negative / accepted costs.**
- `init` signatures in Group A change for callers who never asked for determinism, and the seed is
  required with no default. This is deliberate (D1) and is the single largest source of consumer
  churn in v3.0.0.
- `dir: Io.Dir` adds a parameter at every `ndarray` save/load call site; `Io.Dir.cwd()` is the
  one-token migration.
- Group D is real implementation work (an `Io.Group` rewrite of ten functions), not a rename, and
  is the item most likely to spill past one cycle. It is separable: if it slips, ship it in the
  same major, never after it.
- Anything zuda gains later that owns an OS resource will need `deinit(self, io)`, contradicting
  D3's "`deinit` stays I/O-free". That is a new decision when it happens, not a reason to store
  `io` now.

**Consumer impact (migration issues at release, labels `migration,from:zuda`).**
- `zr` — DAG / Kahn / Levenshtein / glob: unaffected. `WorkStealingDeque.push` gains `io`.
- `silica` — B+Tree, LRU buffer pool, deadlock DFS: unaffected.
- `zoltraak` — `SortedSet` → `SkipList`: `init` gains the seed options struct; HyperLogLog,
  geohash, glob unaffected.
- Any consumer of `NDArray.save`/`load`/`toCSV`/`fromCSV`: gains `io` and `dir`.
