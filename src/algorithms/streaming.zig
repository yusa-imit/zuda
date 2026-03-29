//! Streaming Algorithms Module
//!
//! Algorithms for processing data streams with limited memory.
//! All algorithms work in a single pass over the data with sublinear space.
//!
//! Categories:
//! - Frequency estimation: Count-Min Sketch
//! - Approximate counting: Morris Counter
//! - Cardinality estimation: HyperLogLog
//!
//! Common use cases:
//! - Network monitoring (packet/IP counting)
//! - Database query optimization (selectivity estimation)
//! - Real-time analytics (event frequency tracking)
//! - Memory-constrained environments (IoT, embedded)
//! - Big data processing (distinct value counting)

pub const CountMinSketch = @import("streaming/count_min_sketch.zig").CountMinSketch;
pub const MorrisCounter = @import("streaming/morris_counter.zig").MorrisCounter;
pub const MorrisCounterArray = @import("streaming/morris_counter.zig").MorrisCounterArray;
pub const HyperLogLog = @import("streaming/hyperloglog.zig").HyperLogLog;

test {
    @import("std").testing.refAllDecls(@This());
}
