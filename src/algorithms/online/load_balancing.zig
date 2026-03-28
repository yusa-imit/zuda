//! Online Load Balancing - Assign jobs to machines to minimize makespan
//!
//! Problem: Jobs arrive online and must be assigned to m machines.
//! Goal: Minimize makespan (maximum load on any machine).
//!
//! Competitive Analysis:
//! - Greedy (assign to least loaded): (2 - 1/m)-competitive
//! - Optimal offline: Has perfect knowledge of all jobs
//!
//! Applications:
//! - Task scheduling in distributed systems
//! - Cloud computing resource allocation
//! - Load balancing in web servers
//! - Parallel processing job assignment

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Job to be scheduled
pub const Job = struct {
    id: usize,
    load: f64, // Processing time or resource requirement
};

/// Machine state
pub const Machine = struct {
    id: usize,
    total_load: f64,
    jobs: std.ArrayList(Job),

    /// Initialize machine
    pub fn init(allocator: Allocator, id: usize) Machine {
        return .{
            .id = id,
            .total_load = 0.0,
            .jobs = std.ArrayList(Job).init(allocator),
        };
    }

    /// Clean up resources
    pub fn deinit(self: *Machine) void {
        self.jobs.deinit();
    }

    /// Assign job to this machine
    /// Time: O(1) amortized | Space: O(1)
    pub fn assignJob(self: *Machine, job: Job) !void {
        try self.jobs.append(job);
        self.total_load += job.load;
    }

    /// Get current load
    /// Time: O(1) | Space: O(1)
    pub fn getLoad(self: Machine) f64 {
        return self.total_load;
    }

    /// Get number of jobs
    /// Time: O(1) | Space: O(1)
    pub fn jobCount(self: Machine) usize {
        return self.jobs.items.len;
    }
};

/// Greedy Load Balancing Strategy
/// Assigns each job to the least loaded machine
/// Time: O(n * m) where n is jobs, m is machines | Space: O(n + m)
pub const GreedyLoadBalancer = struct {
    allocator: Allocator,
    machines: std.ArrayList(Machine),
    jobs_assigned: usize,

    /// Initialize with m machines
    /// Time: O(m) | Space: O(m)
    pub fn init(allocator: Allocator, num_machines: usize) !GreedyLoadBalancer {
        var machines = try std.ArrayList(Machine).initCapacity(allocator, num_machines);
        for (0..num_machines) |i| {
            machines.appendAssumeCapacity(Machine.init(allocator, i));
        }

        return GreedyLoadBalancer{
            .allocator = allocator,
            .machines = machines,
            .jobs_assigned = 0,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *GreedyLoadBalancer) void {
        for (self.machines.items) |*machine| {
            machine.deinit();
        }
        self.machines.deinit();
    }

    /// Assign job to least loaded machine
    /// Time: O(m) | Space: O(1)
    pub fn assignJob(self: *GreedyLoadBalancer, job: Job) !usize {
        // Find machine with minimum load
        var min_idx: usize = 0;
        var min_load = self.machines.items[0].getLoad();

        for (self.machines.items[1..], 1..) |machine, i| {
            const load = machine.getLoad();
            if (load < min_load) {
                min_load = load;
                min_idx = i;
            }
        }

        // Assign job
        try self.machines.items[min_idx].assignJob(job);
        self.jobs_assigned += 1;

        return min_idx;
    }

    /// Get makespan (maximum load)
    /// Time: O(m) | Space: O(1)
    pub fn getMakespan(self: GreedyLoadBalancer) f64 {
        var max_load: f64 = 0.0;
        for (self.machines.items) |machine| {
            max_load = @max(max_load, machine.getLoad());
        }
        return max_load;
    }

    /// Get average load
    /// Time: O(m) | Space: O(1)
    pub fn getAverageLoad(self: GreedyLoadBalancer) f64 {
        var total_load: f64 = 0.0;
        for (self.machines.items) |machine| {
            total_load += machine.getLoad();
        }
        return total_load / @as(f64, @floatFromInt(self.machines.items.len));
    }

    /// Get load imbalance (max / avg)
    /// Time: O(m) | Space: O(1)
    pub fn getImbalance(self: GreedyLoadBalancer) f64 {
        const avg = self.getAverageLoad();
        if (avg == 0.0) return 1.0;
        return self.getMakespan() / avg;
    }

    /// Get machine by id
    /// Time: O(1) | Space: O(1)
    pub fn getMachine(self: *GreedyLoadBalancer, machine_id: usize) ?*Machine {
        if (machine_id >= self.machines.items.len) return null;
        return &self.machines.items[machine_id];
    }

    /// Get number of machines
    /// Time: O(1) | Space: O(1)
    pub fn machineCount(self: GreedyLoadBalancer) usize {
        return self.machines.items.len;
    }

    /// Get total jobs assigned
    /// Time: O(1) | Space: O(1)
    pub fn totalJobs(self: GreedyLoadBalancer) usize {
        return self.jobs_assigned;
    }
};

/// Round Robin Load Balancer
/// Assigns jobs in circular order (ignores load)
/// Time: O(n) where n is jobs | Space: O(n + m)
pub const RoundRobinLoadBalancer = struct {
    allocator: Allocator,
    machines: std.ArrayList(Machine),
    next_machine: usize,
    jobs_assigned: usize,

    /// Initialize with m machines
    /// Time: O(m) | Space: O(m)
    pub fn init(allocator: Allocator, num_machines: usize) !RoundRobinLoadBalancer {
        var machines = try std.ArrayList(Machine).initCapacity(allocator, num_machines);
        for (0..num_machines) |i| {
            machines.appendAssumeCapacity(Machine.init(allocator, i));
        }

        return RoundRobinLoadBalancer{
            .allocator = allocator,
            .machines = machines,
            .next_machine = 0,
            .jobs_assigned = 0,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *RoundRobinLoadBalancer) void {
        for (self.machines.items) |*machine| {
            machine.deinit();
        }
        self.machines.deinit();
    }

    /// Assign job to next machine in round-robin order
    /// Time: O(1) amortized | Space: O(1)
    pub fn assignJob(self: *RoundRobinLoadBalancer, job: Job) !usize {
        const machine_id = self.next_machine;
        try self.machines.items[machine_id].assignJob(job);

        self.next_machine = (self.next_machine + 1) % self.machines.items.len;
        self.jobs_assigned += 1;

        return machine_id;
    }

    /// Get makespan (maximum load)
    /// Time: O(m) | Space: O(1)
    pub fn getMakespan(self: RoundRobinLoadBalancer) f64 {
        var max_load: f64 = 0.0;
        for (self.machines.items) |machine| {
            max_load = @max(max_load, machine.getLoad());
        }
        return max_load;
    }

    /// Get average load
    /// Time: O(m) | Space: O(1)
    pub fn getAverageLoad(self: RoundRobinLoadBalancer) f64 {
        var total_load: f64 = 0.0;
        for (self.machines.items) |machine| {
            total_load += machine.getLoad();
        }
        return total_load / @as(f64, @floatFromInt(self.machines.items.len));
    }

    /// Get machine by id
    /// Time: O(1) | Space: O(1)
    pub fn getMachine(self: *RoundRobinLoadBalancer, machine_id: usize) ?*Machine {
        if (machine_id >= self.machines.items.len) return null;
        return &self.machines.items[machine_id];
    }

    /// Get number of machines
    /// Time: O(1) | Space: O(1)
    pub fn machineCount(self: RoundRobinLoadBalancer) usize {
        return self.machines.items.len;
    }
};

/// Compute optimal offline makespan (lower bound)
/// Requires full knowledge of all jobs
/// Time: O(n log n) | Space: O(n)
pub fn optimalOfflineMakespan(allocator: Allocator, jobs: []const Job, num_machines: usize) !f64 {
    // Lower bound: max(max_job, total_load / m)
    var max_job: f64 = 0.0;
    var total_load: f64 = 0.0;

    for (jobs) |job| {
        max_job = @max(max_job, job.load);
        total_load += job.load;
    }

    const avg_load = total_load / @as(f64, @floatFromInt(num_machines));

    // Actual optimal using greedy on sorted jobs (approximation)
    const sorted_jobs = try allocator.dupe(Job, jobs);
    defer allocator.free(sorted_jobs);

    // Sort by load descending
    std.mem.sort(Job, sorted_jobs, {}, struct {
        fn lessThan(_: void, a: Job, b: Job) bool {
            return a.load > b.load;
        }
    }.lessThan);

    var balancer = try GreedyLoadBalancer.init(allocator, num_machines);
    defer balancer.deinit();

    for (sorted_jobs) |job| {
        _ = try balancer.assignJob(job);
    }

    return @max(max_job, @min(avg_load, balancer.getMakespan()));
}

/// Compute competitive ratio
/// Time: O(1) | Space: O(1)
pub fn competitiveRatio(online_makespan: f64, offline_makespan: f64) f64 {
    if (offline_makespan == 0.0) return 1.0;
    return online_makespan / offline_makespan;
}

// ============================================================================
// Tests
// ============================================================================

test "load balancing - greedy: basic assignment" {
    var balancer = try GreedyLoadBalancer.init(testing.allocator, 3);
    defer balancer.deinit();

    const job1 = Job{ .id = 1, .load = 5.0 };
    const job2 = Job{ .id = 2, .load = 3.0 };
    const job3 = Job{ .id = 3, .load = 2.0 };

    _ = try balancer.assignJob(job1);
    _ = try balancer.assignJob(job2);
    _ = try balancer.assignJob(job3);

    try testing.expectEqual(@as(usize, 3), balancer.totalJobs());
    try testing.expectEqual(@as(f64, 5.0), balancer.getMakespan());
}

test "load balancing - greedy: load distribution" {
    var balancer = try GreedyLoadBalancer.init(testing.allocator, 2);
    defer balancer.deinit();

    _ = try balancer.assignJob(.{ .id = 1, .load = 10.0 });
    _ = try balancer.assignJob(.{ .id = 2, .load = 5.0 });
    _ = try balancer.assignJob(.{ .id = 3, .load = 5.0 });

    // Machine 0: 10.0, Machine 1: 5.0 + 5.0 = 10.0
    try testing.expectApproxEqAbs(10.0, balancer.getMakespan(), 0.001);
    try testing.expectApproxEqAbs(10.0, balancer.getAverageLoad(), 0.001);
    try testing.expectApproxEqAbs(1.0, balancer.getImbalance(), 0.001);
}

test "load balancing - greedy: always assigns to least loaded" {
    var balancer = try GreedyLoadBalancer.init(testing.allocator, 3);
    defer balancer.deinit();

    const m0 = try balancer.assignJob(.{ .id = 1, .load = 10.0 });
    const m1 = try balancer.assignJob(.{ .id = 2, .load = 5.0 });
    const m2 = try balancer.assignJob(.{ .id = 3, .load = 3.0 });

    // All different machines initially
    try testing.expect(m0 != m1);
    try testing.expect(m1 != m2);

    // Next job should go to machine with 3.0 load
    const m3 = try balancer.assignJob(.{ .id = 4, .load = 2.0 });
    try testing.expectEqual(m2, m3);
}

test "load balancing - greedy: imbalance calculation" {
    var balancer = try GreedyLoadBalancer.init(testing.allocator, 3);
    defer balancer.deinit();

    _ = try balancer.assignJob(.{ .id = 1, .load = 10.0 });
    _ = try balancer.assignJob(.{ .id = 2, .load = 1.0 });
    _ = try balancer.assignJob(.{ .id = 3, .load = 1.0 });

    // Makespan: 10.0, Average: 4.0, Imbalance: 2.5
    try testing.expectApproxEqAbs(10.0, balancer.getMakespan(), 0.001);
    try testing.expectApproxEqAbs(4.0, balancer.getAverageLoad(), 0.001);
    try testing.expectApproxEqAbs(2.5, balancer.getImbalance(), 0.001);
}

test "load balancing - greedy: machine access" {
    var balancer = try GreedyLoadBalancer.init(testing.allocator, 2);
    defer balancer.deinit();

    const m0 = try balancer.assignJob(.{ .id = 1, .load = 5.0 });

    const machine = balancer.getMachine(m0).?;
    try testing.expectEqual(@as(usize, 1), machine.jobCount());
    try testing.expectApproxEqAbs(5.0, machine.getLoad(), 0.001);
}

test "load balancing - round robin: basic assignment" {
    var balancer = try RoundRobinLoadBalancer.init(testing.allocator, 3);
    defer balancer.deinit();

    const m0 = try balancer.assignJob(.{ .id = 1, .load = 10.0 });
    const m1 = try balancer.assignJob(.{ .id = 2, .load = 10.0 });
    const m2 = try balancer.assignJob(.{ .id = 3, .load = 10.0 });
    const m3 = try balancer.assignJob(.{ .id = 4, .load = 10.0 });

    // Should cycle: 0, 1, 2, 0
    try testing.expectEqual(@as(usize, 0), m0);
    try testing.expectEqual(@as(usize, 1), m1);
    try testing.expectEqual(@as(usize, 2), m2);
    try testing.expectEqual(@as(usize, 0), m3);
}

test "load balancing - round robin: ignores load" {
    var balancer = try RoundRobinLoadBalancer.init(testing.allocator, 2);
    defer balancer.deinit();

    _ = try balancer.assignJob(.{ .id = 1, .load = 100.0 }); // Machine 0
    _ = try balancer.assignJob(.{ .id = 2, .load = 1.0 }); // Machine 1
    _ = try balancer.assignJob(.{ .id = 3, .load = 1.0 }); // Machine 0 (not greedy!)

    const m0 = balancer.getMachine(0).?;
    const m1 = balancer.getMachine(1).?;

    try testing.expectApproxEqAbs(101.0, m0.getLoad(), 0.001);
    try testing.expectApproxEqAbs(1.0, m1.getLoad(), 0.001);
}

test "load balancing - optimal offline: lower bound" {
    const jobs = [_]Job{
        .{ .id = 1, .load = 10.0 },
        .{ .id = 2, .load = 5.0 },
        .{ .id = 3, .load = 5.0 },
    };

    const makespan = try optimalOfflineMakespan(testing.allocator, &jobs, 2);

    // Optimal: Machine 0: 10.0, Machine 1: 5.0 + 5.0 = 10.0
    try testing.expectApproxEqAbs(10.0, makespan, 0.001);
}

test "load balancing - competitive ratio: greedy vs optimal" {
    const jobs = [_]Job{
        .{ .id = 1, .load = 10.0 },
        .{ .id = 2, .load = 10.0 },
        .{ .id = 3, .load = 10.0 },
        .{ .id = 4, .load = 1.0 },
    };

    var greedy = try GreedyLoadBalancer.init(testing.allocator, 3);
    defer greedy.deinit();

    for (jobs) |job| {
        _ = try greedy.assignJob(job);
    }

    const online_makespan = greedy.getMakespan();
    const offline_makespan = try optimalOfflineMakespan(testing.allocator, &jobs, 3);

    const ratio = competitiveRatio(online_makespan, offline_makespan);

    // Greedy is (2 - 1/m)-competitive, m=3 → 5/3 ≈ 1.67
    try testing.expect(ratio <= 2.0);
    try testing.expect(ratio >= 1.0);
}

test "load balancing - greedy better than round robin" {
    const jobs = [_]Job{
        .{ .id = 1, .load = 100.0 },
        .{ .id = 2, .load = 1.0 },
        .{ .id = 3, .load = 1.0 },
        .{ .id = 4, .load = 1.0 },
    };

    var greedy = try GreedyLoadBalancer.init(testing.allocator, 2);
    defer greedy.deinit();

    var rr = try RoundRobinLoadBalancer.init(testing.allocator, 2);
    defer rr.deinit();

    for (jobs) |job| {
        _ = try greedy.assignJob(job);
        _ = try rr.assignJob(job);
    }

    // Greedy: 100, 3 → better balance
    // RR: 101, 2 → worse balance
    try testing.expect(greedy.getMakespan() <= rr.getMakespan());
}

test "load balancing - many machines" {
    var balancer = try GreedyLoadBalancer.init(testing.allocator, 10);
    defer balancer.deinit();

    for (0..100) |i| {
        _ = try balancer.assignJob(.{ .id = i, .load = @floatFromInt(i % 10 + 1) });
    }

    try testing.expectEqual(@as(usize, 100), balancer.totalJobs());
    try testing.expect(balancer.getMakespan() > 0.0);
}

test "load balancing - memory safety" {
    var greedy = try GreedyLoadBalancer.init(testing.allocator, 5);
    defer greedy.deinit();

    for (0..1000) |i| {
        _ = try greedy.assignJob(.{ .id = i, .load = @floatFromInt(i % 100 + 1) });
    }

    var rr = try RoundRobinLoadBalancer.init(testing.allocator, 5);
    defer rr.deinit();

    for (0..1000) |i| {
        _ = try rr.assignJob(.{ .id = i, .load = @floatFromInt(i % 100 + 1) });
    }
}
