//! Geohash encoding and decoding for geospatial indexing.
//!
//! Geohash is a hierarchical spatial data structure which subdivides space into
//! buckets of grid shape. It provides a geocode system where nearby locations
//! share similar hash prefixes, making it useful for proximity searches.
//!
//! Time complexity: O(precision) for encode/decode
//! Space complexity: O(precision)
//!
//! Reference: https://en.wikipedia.org/wiki/Geohash

const std = @import("std");

/// Base32 alphabet used in geohash encoding (excludes a, i, l, o to avoid confusion)
const BASE32: [32]u8 = "0123456789bcdefghjkmnpqrstuvwxyz".*;

/// Bounding box representing a geographic area
pub const BoundingBox = struct {
    min_lat: f64,
    max_lat: f64,
    min_lon: f64,
    max_lon: f64,

    /// Get the center point of the bounding box
    /// Time: O(1) | Space: O(1)
    pub fn center(self: BoundingBox) struct { lat: f64, lon: f64 } {
        return .{
            .lat = (self.min_lat + self.max_lat) / 2.0,
            .lon = (self.min_lon + self.max_lon) / 2.0,
        };
    }

    /// Get the width (longitude span) of the bounding box
    /// Time: O(1) | Space: O(1)
    pub fn width(self: BoundingBox) f64 {
        return self.max_lon - self.min_lon;
    }

    /// Get the height (latitude span) of the bounding box
    /// Time: O(1) | Space: O(1)
    pub fn height(self: BoundingBox) f64 {
        return self.max_lat - self.min_lat;
    }
};

/// Encode a latitude and longitude into a geohash string.
/// Time: O(precision) | Space: O(precision)
///
/// Arguments:
/// - lat: latitude in degrees [-90, 90]
/// - lon: longitude in degrees [-180, 180]
/// - precision: length of output geohash (1-12 recommended)
/// - buffer: output buffer, must have at least `precision` bytes
///
/// Returns: slice of buffer containing the geohash
pub fn encode(lat: f64, lon: f64, precision: usize, buffer: []u8) ![]const u8 {
    if (precision == 0 or precision > buffer.len) {
        return error.InvalidPrecision;
    }
    if (lat < -90.0 or lat > 90.0) {
        return error.InvalidLatitude;
    }
    if (lon < -180.0 or lon > 180.0) {
        return error.InvalidLongitude;
    }

    var lat_range = [2]f64{ -90.0, 90.0 };
    var lon_range = [2]f64{ -180.0, 180.0 };
    var is_even = true;
    var bit: u5 = 0;
    var ch: usize = 0;
    var idx: usize = 0;

    while (idx < precision) {
        if (is_even) {
            const mid = (lon_range[0] + lon_range[1]) / 2.0;
            if (lon > mid) {
                ch |= (@as(usize, 1) << @intCast(4 - bit));
                lon_range[0] = mid;
            } else {
                lon_range[1] = mid;
            }
        } else {
            const mid = (lat_range[0] + lat_range[1]) / 2.0;
            if (lat > mid) {
                ch |= (@as(usize, 1) << @intCast(4 - bit));
                lat_range[0] = mid;
            } else {
                lat_range[1] = mid;
            }
        }

        is_even = !is_even;

        if (bit == 4) {
            buffer[idx] = BASE32[ch];
            idx += 1;
            bit = 0;
            ch = 0;
        } else {
            bit += 1;
        }
    }

    return buffer[0..precision];
}

/// Decode a geohash string into a bounding box.
/// Time: O(len(hash)) | Space: O(1)
///
/// Returns: bounding box containing the area represented by the geohash
pub fn decode(hash: []const u8) !BoundingBox {
    if (hash.len == 0) {
        return error.EmptyHash;
    }

    var lat_range = [2]f64{ -90.0, 90.0 };
    var lon_range = [2]f64{ -180.0, 180.0 };
    var is_even = true;

    for (hash) |c| {
        const idx = charToIndex(c) catch return error.InvalidCharacter;
        var mask: u5 = 16; // 0b10000

        while (mask > 0) : (mask >>= 1) {
            if (is_even) {
                const mid = (lon_range[0] + lon_range[1]) / 2.0;
                if ((idx & mask) != 0) {
                    lon_range[0] = mid;
                } else {
                    lon_range[1] = mid;
                }
            } else {
                const mid = (lat_range[0] + lat_range[1]) / 2.0;
                if ((idx & mask) != 0) {
                    lat_range[0] = mid;
                } else {
                    lat_range[1] = mid;
                }
            }
            is_even = !is_even;
        }
    }

    return BoundingBox{
        .min_lat = lat_range[0],
        .max_lat = lat_range[1],
        .min_lon = lon_range[0],
        .max_lon = lon_range[1],
    };
}

/// Get the 8 adjacent geohashes (neighbors) of a given geohash.
/// Time: O(precision) | Space: O(precision)
///
/// The order is: N, NE, E, SE, S, SW, W, NW
///
/// Arguments:
/// - hash: input geohash
/// - buffers: array of 8 buffers, each must have at least hash.len bytes
///
/// Returns: array of 8 neighboring geohashes (or errors)
pub fn neighbors(hash: []const u8, buffers: *[8][]u8) ![8][]const u8 {
    const directions = [_]Direction{ .north, .north_east, .east, .south_east, .south, .south_west, .west, .north_west };
    var result: [8][]const u8 = undefined;

    for (directions, 0..) |dir, i| {
        result[i] = try neighbor(hash, dir, buffers[i]);
    }

    return result;
}

/// Direction enum for neighbor calculation
pub const Direction = enum {
    north,
    south,
    east,
    west,
    north_east,
    north_west,
    south_east,
    south_west,
};

/// Get a neighboring geohash in a specific direction.
/// Time: O(precision) | Space: O(precision)
pub fn neighbor(hash: []const u8, direction: Direction, buffer: []u8) ![]const u8 {
    if (hash.len == 0) return error.EmptyHash;
    if (buffer.len < hash.len) return error.BufferTooSmall;

    const bbox = try decode(hash);
    const center_point = bbox.center();
    const lat_delta = bbox.height();
    const lon_delta = bbox.width();

    var new_lat = center_point.lat;
    var new_lon = center_point.lon;

    switch (direction) {
        .north => new_lat += lat_delta,
        .south => new_lat -= lat_delta,
        .east => new_lon += lon_delta,
        .west => new_lon -= lon_delta,
        .north_east => {
            new_lat += lat_delta;
            new_lon += lon_delta;
        },
        .north_west => {
            new_lat += lat_delta;
            new_lon -= lon_delta;
        },
        .south_east => {
            new_lat -= lat_delta;
            new_lon += lon_delta;
        },
        .south_west => {
            new_lat -= lat_delta;
            new_lon -= lon_delta;
        },
    }

    // Wrap longitude
    if (new_lon > 180.0) new_lon -= 360.0;
    if (new_lon < -180.0) new_lon += 360.0;

    // Clamp latitude
    if (new_lat > 90.0) new_lat = 90.0;
    if (new_lat < -90.0) new_lat = -90.0;

    return encode(new_lat, new_lon, hash.len, buffer);
}

/// Convert a base32 character to its index value (0-31)
fn charToIndex(c: u8) !u5 {
    return switch (c) {
        '0'...'9' => @intCast(c - '0'),
        'b' => 10,
        'c' => 11,
        'd' => 12,
        'e' => 13,
        'f' => 14,
        'g' => 15,
        'h' => 16,
        'j' => 17,
        'k' => 18,
        'm' => 19,
        'n' => 20,
        'p' => 21,
        'q' => 22,
        'r' => 23,
        's' => 24,
        't' => 25,
        'u' => 26,
        'v' => 27,
        'w' => 28,
        'x' => 29,
        'y' => 30,
        'z' => 31,
        else => error.InvalidCharacter,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "geohash: basic encoding" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;

    // San Francisco - just verify encoding works and produces valid base32
    const hash = try encode(37.7749, -122.4194, 9, &buffer);
    try testing.expect(hash.len == 9);

    // Verify all characters are valid base32
    for (hash) |c| {
        _ = try charToIndex(c);
    }
}

test "geohash: encoding precision levels" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;
    const lat = 51.5074;
    const lon = -0.1278;

    const p1 = try encode(lat, lon, 1, &buffer);
    try testing.expect(p1.len == 1);

    const p5 = try encode(lat, lon, 5, &buffer);
    try testing.expect(p5.len == 5);

    const p9 = try encode(lat, lon, 9, &buffer);
    try testing.expect(p9.len == 9);

    // Longer precision should start with shorter precision as prefix
    try testing.expect(std.mem.startsWith(u8, p5, p1));
    try testing.expect(std.mem.startsWith(u8, p9, p5));
}

test "geohash: basic decoding" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;
    const lat = 37.7749;
    const lon = -122.4194;

    const hash = try encode(lat, lon, 9, &buffer);
    const bbox = try decode(hash);

    // Check that San Francisco is within the decoded bounding box
    try testing.expect(bbox.min_lat <= lat and bbox.max_lat >= lat);
    try testing.expect(bbox.min_lon <= lon and bbox.max_lon >= lon);
}

test "geohash: encode-decode round trip" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;
    const lat = 37.7749;
    const lon = -122.4194;

    const hash = try encode(lat, lon, 8, &buffer);
    const bbox = try decode(hash);
    const center_point = bbox.center();

    // Center should be close to original coordinates
    try testing.expectApproxEqAbs(lat, center_point.lat, 0.001);
    try testing.expectApproxEqAbs(lon, center_point.lon, 0.001);
}

test "geohash: precision affects accuracy" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;
    const lat = 40.7128;
    const lon = -74.0060;

    const hash3 = try encode(lat, lon, 3, &buffer);
    const bbox3 = try decode(hash3);

    const hash6 = try encode(lat, lon, 6, &buffer);
    const bbox6 = try decode(hash6);

    const hash9 = try encode(lat, lon, 9, &buffer);
    const bbox9 = try decode(hash9);

    // Higher precision = smaller bounding box
    try testing.expect(bbox3.width() > bbox6.width());
    try testing.expect(bbox6.width() > bbox9.width());
    try testing.expect(bbox3.height() > bbox6.height());
    try testing.expect(bbox6.height() > bbox9.height());
}

test "geohash: invalid coordinates" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;

    // Invalid latitude
    try testing.expectError(error.InvalidLatitude, encode(91.0, 0.0, 5, &buffer));
    try testing.expectError(error.InvalidLatitude, encode(-91.0, 0.0, 5, &buffer));

    // Invalid longitude
    try testing.expectError(error.InvalidLongitude, encode(0.0, 181.0, 5, &buffer));
    try testing.expectError(error.InvalidLongitude, encode(0.0, -181.0, 5, &buffer));

    // Invalid precision
    try testing.expectError(error.InvalidPrecision, encode(0.0, 0.0, 0, &buffer));
    try testing.expectError(error.InvalidPrecision, encode(0.0, 0.0, 20, &buffer));
}

test "geohash: invalid hash characters" {
    const testing = std.testing;

    // 'a', 'i', 'l', 'o' are not in base32 alphabet
    try testing.expectError(error.InvalidCharacter, decode("abc"));
    try testing.expectError(error.InvalidCharacter, decode("9q8yik"));
    try testing.expectError(error.InvalidCharacter, decode("u10hbl"));
    try testing.expectError(error.InvalidCharacter, decode("9qo"));

    // Empty hash
    try testing.expectError(error.EmptyHash, decode(""));
}

test "geohash: neighbor calculation" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;
    var neighbor_buf: [12]u8 = undefined;

    const hash = try encode(37.7749, -122.4194, 6, &buffer);

    // North neighbor
    const north = try neighbor(hash, .north, &neighbor_buf);
    const north_bbox = try decode(north);
    const original_bbox = try decode(hash);

    // North neighbor should have higher latitude
    try testing.expect(north_bbox.min_lat > original_bbox.min_lat);
}

test "geohash: all 8 neighbors" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;
    const hash = try encode(0.0, 0.0, 5, &buffer);

    var buffers: [8][]u8 = undefined;
    var storage: [8][12]u8 = undefined;
    for (&storage, 0..) |*buf, i| {
        buffers[i] = buf;
    }

    const nbrs = try neighbors(hash, &buffers);

    // All neighbors should be non-empty and different from original
    for (nbrs) |nbr| {
        try testing.expect(nbr.len > 0);
        try testing.expect(!std.mem.eql(u8, nbr, hash));
    }

    // All neighbors should be unique
    for (nbrs, 0..) |nbr1, i| {
        for (nbrs[i + 1 ..]) |nbr2| {
            try testing.expect(!std.mem.eql(u8, nbr1, nbr2));
        }
    }
}

test "geohash: prefix similarity for nearby points" {
    const testing = std.testing;

    var buf1: [12]u8 = undefined;
    var buf2: [12]u8 = undefined;

    // Two nearby points in San Francisco
    const hash1 = try encode(37.7749, -122.4194, 8, &buf1);
    const hash2 = try encode(37.7750, -122.4195, 8, &buf2);

    // They should share a common prefix
    var common_len: usize = 0;
    for (hash1, hash2) |c1, c2| {
        if (c1 == c2) {
            common_len += 1;
        } else {
            break;
        }
    }

    try testing.expect(common_len >= 6);
}

test "geohash: equator and prime meridian" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;

    const hash = try encode(0.0, 0.0, 5, &buffer);
    const bbox = try decode(hash);
    const center_point = bbox.center();

    try testing.expectApproxEqAbs(0.0, center_point.lat, 0.1);
    try testing.expectApproxEqAbs(0.0, center_point.lon, 0.1);
}

test "geohash: extreme coordinates" {
    const testing = std.testing;

    var buffer: [12]u8 = undefined;

    // North pole
    const north_hash = try encode(90.0, 0.0, 5, &buffer);
    const north_bbox = try decode(north_hash);
    try testing.expect(north_bbox.max_lat == 90.0);

    // South pole
    const south_hash = try encode(-90.0, 0.0, 5, &buffer);
    const south_bbox = try decode(south_hash);
    try testing.expect(south_bbox.min_lat == -90.0);

    // Date line
    const east_hash = try encode(0.0, 180.0, 5, &buffer);
    const east_bbox = try decode(east_hash);
    try testing.expect(east_bbox.max_lon == 180.0);

    const west_hash = try encode(0.0, -180.0, 5, &buffer);
    const west_bbox = try decode(west_hash);
    try testing.expect(west_bbox.min_lon == -180.0);
}

test "geohash: bounding box utilities" {
    const testing = std.testing;

    const bbox = BoundingBox{
        .min_lat = 30.0,
        .max_lat = 40.0,
        .min_lon = -100.0,
        .max_lon = -90.0,
    };

    const center_point = bbox.center();
    try testing.expectApproxEqAbs(35.0, center_point.lat, 0.001);
    try testing.expectApproxEqAbs(-95.0, center_point.lon, 0.001);

    try testing.expectApproxEqAbs(10.0, bbox.width(), 0.001);
    try testing.expectApproxEqAbs(10.0, bbox.height(), 0.001);
}

test "geohash: stress test with random coordinates" {
    const testing = std.testing;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var buffer: [12]u8 = undefined;

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const lat = random.float(f64) * 180.0 - 90.0;
        const lon = random.float(f64) * 360.0 - 180.0;
        const precision = random.intRangeAtMost(usize, 1, 10);

        const hash = try encode(lat, lon, precision, &buffer);
        try testing.expect(hash.len == precision);

        const bbox = try decode(hash);
        try testing.expect(bbox.min_lat <= lat and bbox.max_lat >= lat);
        try testing.expect(bbox.min_lon <= lon and bbox.max_lon >= lon);
    }
}
