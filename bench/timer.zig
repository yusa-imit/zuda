//! Wall-clock stopwatch for the benchmark executables.
//!
//! Replaces the removed `std.time.Timer`: the executable's `main` owns the `std.Io` it was
//! given and passes it here, so the clock is injected rather than read globally.
//! Allocation: none. Not thread-safe; one `Timer` per measuring thread.

const std = @import("std");

pub const Timer = struct {
    io: std.Io,
    start_ns: i96,

    /// Start a stopwatch on the monotonic clock of `io`.
    pub fn start(io: std.Io) Timer {
        return .{ .io = io, .start_ns = now_ns(io) };
    }

    /// Nanoseconds since `start` or the last `reset`.
    pub fn read(timer: *const Timer) u64 {
        const elapsed_ns = now_ns(timer.io) - timer.start_ns;
        std.debug.assert(elapsed_ns >= 0);
        return @intCast(elapsed_ns);
    }

    /// Restart the stopwatch from now.
    pub fn reset(timer: *Timer) void {
        timer.start_ns = now_ns(timer.io);
    }

    fn now_ns(io: std.Io) i96 {
        return std.Io.Clock.awake.now(io).nanoseconds;
    }
};

test "Timer.read is monotonic and reset restarts it" {
    var timer = Timer.start(std.testing.io);
    const first_ns = timer.read();
    const second_ns = timer.read();
    try std.testing.expect(second_ns >= first_ns);

    timer.reset();
    try std.testing.expect(timer.read() <= second_ns + 1_000_000_000);
}
