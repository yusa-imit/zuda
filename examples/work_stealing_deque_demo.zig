const std = @import("std");
const zuda = @import("zuda");

const WorkStealingDeque = zuda.containers.queues.WorkStealingDeque;

/// Work-Stealing Deque API Demonstration
///
/// Demonstrates the Chase-Lev work-stealing deque used for parallel task execution.
///
/// **Consumer Use Case**: zr task runner (replaces src/exec/workstealing.zig — 130 LOC)
/// Current: Custom 130-line implementation
/// With zuda: @import("zuda").containers.queues.work_stealing_deque.WorkStealingDeque
/// Advantages: Lock-free stealing, cache-friendly owner operations, proven algorithm
///
/// **Key Concepts**:
/// - Owner thread: push/pop from bottom (LIFO for cache locality)
/// - Stealer threads: steal from top (FIFO for load balancing)
/// - Lock-free: Common case (pop/steal) requires no locks
/// - Resize: Protected by mutex, rare due to amortized growth
///
/// Run: zig build example-work-stealing-deque

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Work-Stealing Deque API Demo ===\n\n", .{});

    try demo1_basic_operations(allocator);
    try demo2_lifo_fifo_behavior(allocator);
    try demo3_parallel_work_stealing(allocator);
    try demo4_task_queue_simulation(allocator);

    std.debug.print("\n=== API Summary ===\n", .{});
    std.debug.print("• init(allocator) → !Self             — Create deque (O(1))\n", .{});
    std.debug.print("• push(task) → !void                  — Owner adds task to bottom (O(1) amortized)\n", .{});
    std.debug.print("• pop() → ?T                          — Owner removes from bottom, LIFO (O(1))\n", .{});
    std.debug.print("• steal() → ?T                        — Stealer removes from top, FIFO (O(1))\n", .{});
    std.debug.print("• size() → usize                      — Approximate size (O(1))\n", .{});
    std.debug.print("• isEmpty() → bool                    — Check if empty (O(1))\n", .{});
    std.debug.print("\n**Consumer**: zr task runner — replaces 130-line custom implementation\n", .{});
    std.debug.print("**Algorithm**: Chase-Lev lock-free work-stealing deque\n", .{});
}

/// Demo 1: Basic push/pop operations (owner thread)
fn demo1_basic_operations(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Demo 1: Basic Operations ---\n", .{});

    var deque = try WorkStealingDeque(u32).init(allocator);
    defer deque.deinit();

    // Owner pushes tasks
    try deque.push(100);
    try deque.push(200);
    try deque.push(300);
    std.debug.print("After push(100, 200, 300): size={d}\n", .{deque.size()});

    // Owner pops (LIFO: last in, first out for cache locality)
    const task1 = deque.pop();
    const task2 = deque.pop();
    std.debug.print("Owner pop: {?d}, {?d} (LIFO order: 300, 200)\n", .{ task1, task2 });
    std.debug.print("Remaining size: {d}\n", .{deque.size()});

    // Pop last item
    const task3 = deque.pop();
    std.debug.print("Owner pop: {?d} (last item: 100)\n", .{task3});
    std.debug.print("isEmpty: {}\n\n", .{deque.isEmpty()});
}

/// Demo 2: LIFO (owner) vs FIFO (stealer) behavior
fn demo2_lifo_fifo_behavior(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Demo 2: LIFO (Owner) vs FIFO (Stealer) ---\n", .{});

    var deque = try WorkStealingDeque(u32).init(allocator);
    defer deque.deinit();

    // Owner fills deque
    try deque.push(1);
    try deque.push(2);
    try deque.push(3);
    try deque.push(4);
    try deque.push(5);
    std.debug.print("Deque: [1, 2, 3, 4, 5] (bottom→1, top→5)\n", .{});

    // Owner pops (LIFO from bottom)
    std.debug.print("\nOwner behavior (LIFO from bottom):\n", .{});
    std.debug.print("  pop() → {?d}  (most recent: 5)\n", .{deque.pop()});
    std.debug.print("  pop() → {?d}  (next recent: 4)\n", .{deque.pop()});

    // Stealer steals (FIFO from top)
    std.debug.print("\nStealer behavior (FIFO from top):\n", .{});
    std.debug.print("  steal() → {?d}  (oldest: 1)\n", .{deque.steal()});
    std.debug.print("  steal() → {?d}  (next oldest: 2)\n", .{deque.steal()});

    // Last item: race between owner pop and stealer steal
    std.debug.print("\nLast item (race condition):\n", .{});
    const owner_result = deque.pop();
    const stealer_result = deque.steal();
    std.debug.print("  Owner pop: {?d}\n", .{owner_result});
    std.debug.print("  Stealer steal: {?d}\n", .{stealer_result});
    std.debug.print("  (Exactly one succeeds, the other gets null)\n\n", .{});
}

/// Demo 3: Parallel work stealing with threads
fn demo3_parallel_work_stealing(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Demo 3: Parallel Work Stealing ---\n", .{});

    const Context = struct {
        deque: *WorkStealingDeque(u32),
        stolen: *std.ArrayList(u32),
        mutex: *std.Thread.Mutex,
        allocator: std.mem.Allocator,
    };

    var deque = try WorkStealingDeque(u32).init(allocator);
    defer deque.deinit();

    // Owner fills deque with tasks
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        try deque.push(i);
    }
    std.debug.print("Owner created 20 tasks [0..19]\n", .{});

    // Stealer thread function
    const stealer_fn = struct {
        fn run(ctx: Context) void {
            var count: u32 = 0;
            while (ctx.deque.steal()) |task| {
                count += 1;
                ctx.mutex.lock();
                ctx.stolen.append(ctx.allocator, task) catch {};
                ctx.mutex.unlock();
                // Simulate work
                std.Thread.sleep(100_000); // 100µs
            }
        }
    }.run;

    var stolen = try std.ArrayList(u32).initCapacity(allocator, 20);
    defer stolen.deinit(allocator);
    var mutex = std.Thread.Mutex{};

    const ctx = Context{ .deque = &deque, .stolen = &stolen, .mutex = &mutex, .allocator = allocator };

    // Spawn 2 stealer threads
    const thread1 = try std.Thread.spawn(.{}, stealer_fn, .{ctx});
    const thread2 = try std.Thread.spawn(.{}, stealer_fn, .{ctx});

    // Owner pops some tasks (LIFO) while stealers steal (FIFO)
    var owner_count: u32 = 0;
    var owner_tasks = try std.ArrayList(u32).initCapacity(allocator, 20);
    defer owner_tasks.deinit(allocator);

    while (deque.pop()) |task| {
        owner_count += 1;
        try owner_tasks.append(allocator, task);
        std.Thread.sleep(150_000); // 150µs (slower than stealers)
    }

    // Wait for stealers to finish
    thread1.join();
    thread2.join();

    std.debug.print("Owner processed: {d} tasks (LIFO: recent tasks)\n", .{owner_count});
    std.debug.print("  Owner got: ", .{});
    for (owner_tasks.items[0..@min(5, owner_tasks.items.len)]) |task| {
        std.debug.print("{d} ", .{task});
    }
    if (owner_tasks.items.len > 5) std.debug.print("...", .{});
    std.debug.print("\n", .{});

    std.debug.print("Stealers processed: {d} tasks (FIFO: old tasks)\n", .{stolen.items.len});
    std.debug.print("  Stealers got: ", .{});
    for (stolen.items[0..@min(5, stolen.items.len)]) |task| {
        std.debug.print("{d} ", .{task});
    }
    if (stolen.items.len > 5) std.debug.print("...", .{});
    std.debug.print("\n", .{});

    std.debug.print("Total: {d} tasks (owner {d} + stealers {d})\n", .{ owner_count + @as(u32, @intCast(stolen.items.len)), owner_count, stolen.items.len });
    std.debug.print("Result: Load balanced — owner kept recent (cache-warm) tasks, stealers got old tasks\n\n", .{});
}

/// Demo 4: Task queue simulation (zr use case)
fn demo4_task_queue_simulation(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Demo 4: Task Queue Simulation (zr Use Case) ---\n", .{});

    const Task = struct {
        id: u32,
        name: []const u8,
    };

    var deque = try WorkStealingDeque(Task).init(allocator);
    defer deque.deinit();

    // Owner thread (main task scheduler) pushes tasks
    try deque.push(.{ .id = 1, .name = "compile_main.zig" });
    try deque.push(.{ .id = 2, .name = "compile_utils.zig" });
    try deque.push(.{ .id = 3, .name = "link_binary" });
    try deque.push(.{ .id = 4, .name = "run_tests" });
    std.debug.print("Main scheduler: Pushed 4 tasks to deque\n", .{});

    // Worker 1 (owner) processes recent tasks (LIFO for cache locality)
    std.debug.print("\nWorker 1 (owner) — LIFO for cache locality:\n", .{});
    if (deque.pop()) |task| {
        std.debug.print("  Executing: {s} (id={d}) [most recent, likely cache-hot]\n", .{ task.name, task.id });
    }
    if (deque.pop()) |task| {
        std.debug.print("  Executing: {s} (id={d}) [next recent]\n", .{ task.name, task.id });
    }

    // Worker 2 (stealer) steals old tasks (FIFO for load balancing)
    std.debug.print("\nWorker 2 (stealer) — FIFO for load balancing:\n", .{});
    if (deque.steal()) |task| {
        std.debug.print("  Stole: {s} (id={d}) [oldest task, helps owner]\n", .{ task.name, task.id });
    }

    // Owner finishes remaining work
    std.debug.print("\nWorker 1 (owner) — Finishing:\n", .{});
    while (deque.pop()) |task| {
        std.debug.print("  Executing: {s} (id={d})\n", .{ task.name, task.id });
    }

    std.debug.print("\nAll tasks completed!\n", .{});
    std.debug.print("**zr benefit**: Replace 130-line custom impl with zuda WorkStealingDeque\n", .{});
    std.debug.print("  • Proven Chase-Lev algorithm (used in Java ForkJoinPool, Cilk, Go scheduler)\n", .{});
    std.debug.print("  • Lock-free stealing (no contention in common case)\n", .{});
    std.debug.print("  • LIFO owner pops (cache-friendly for recent tasks)\n", .{});
    std.debug.print("  • FIFO stealer steals (load balancing, take old tasks)\n", .{});
    std.debug.print("  • Memory-efficient dynamic resizing\n", .{});
}
