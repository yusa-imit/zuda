/// Compression Algorithms
///
/// This module provides fundamental lossless compression algorithms used in
/// file compression, data transmission, and storage optimization.
///
/// ## Algorithms Overview
///
/// ### Run-Length Encoding (RLE)
/// **Use Case**: Data with long runs of identical values
/// - **Best For**: Simple images, fax transmissions, DNA sequences
/// - **Time**: O(n) encode/decode
/// - **Space**: O(k) where k = number of runs
/// - **Pros**: Simple, fast, works well on repetitive data
/// - **Cons**: Can expand data if no runs exist
///
/// ### Delta Encoding
/// **Use Case**: Sequential numerical data with small changes
/// - **Best For**: Time series, sensor data, audio samples
/// - **Time**: O(n) encode/decode
/// - **Space**: O(n) - same size but smaller values
/// - **Pros**: Reduces magnitude of values, enables better entropy coding
/// - **Cons**: No compression alone, requires follow-up compression
/// - **Variants**: First-order (differences), second-order (delta of deltas)
///
/// ### LZ77
/// **Use Case**: General-purpose dictionary-based compression
/// - **Best For**: Text files, general data (used in gzip, PNG/DEFLATE)
/// - **Time**: O(n × w) where w = window size
/// - **Space**: O(n) output, O(w) for window
/// - **Pros**: No prior knowledge needed, good compression ratios
/// - **Cons**: Slower than RLE, sensitive to window size
/// - **Foundation**: Base for gzip, PNG, DEFLATE algorithm
///
/// ### LZSS (Lempel-Ziv-Storer-Szymanski)
/// **Use Case**: Improved LZ77 with flag bits for efficiency
/// - **Best For**: General-purpose compression, embedded systems (ARJ, LHA)
/// - **Time**: O(n × w) encode, O(m) decode where w = window size, m = output
/// - **Space**: O(w) for window, typically 40-60% of input for text
/// - **Pros**: Better than LZ77 (no overhead for literals), simple decode
/// - **Cons**: Slightly more complex encoding (flag bit management)
/// - **Note**: Uses 1-bit flags to distinguish literals from (offset, length) references
///
/// ### Burrows-Wheeler Transform (BWT)
/// **Use Case**: Pre-processing for better compression
/// - **Best For**: Text data (used in bzip2)
/// - **Time**: O(n² log n) naive, O(n) with suffix array
/// - **Space**: O(n²) naive, O(n) with suffix array
/// - **Pros**: Groups similar characters, excellent with RLE/MTF
/// - **Cons**: Computationally expensive, needs follow-up compression
/// - **Note**: Not compression itself, but enables better compression
///
/// ### Huffman Coding
/// **Use Case**: Optimal prefix-free entropy encoding
/// - **Best For**: General compression (ZIP/GZIP, JPEG, PNG, MP3)
/// - **Time**: O(n log k) encode, O(m) decode where n = data length, k = alphabet size, m = encoded bits
/// - **Space**: O(k) for tree and codebook
/// - **Pros**: Simple, fast, optimal for symbol-wise encoding
/// - **Cons**: Uses whole bits (less optimal than arithmetic for skewed data)
/// - **Note**: Foundation for DEFLATE (LZ77 + Huffman), used in ZIP, GZIP, PNG
///
/// ### Arithmetic Coding
/// **Use Case**: Statistical entropy-based compression
/// - **Best For**: Highly skewed probability distributions (JPEG 2000, H.264/H.265)
/// - **Time**: O(n × k) where k = alphabet size (256 for bytes)
/// - **Space**: O(k) for frequency table + O(n) for output
/// - **Pros**: Better than Huffman for skewed data, fractional bits per symbol
/// - **Cons**: More complex, requires high-precision arithmetic
/// - **Note**: Achieves compression closer to theoretical entropy limit
///
/// ### LZ4
/// **Use Case**: Fast compression/decompression for real-time systems
/// - **Best For**: Databases (RocksDB, MongoDB), filesystems (ZFS, Btrfs), network protocols
/// - **Time**: O(n) compression, O(m) decompression (m = output length)
/// - **Space**: O(hash_table + output)
/// - **Pros**: Very fast decompression (~2000 MB/s), simple algorithm, streaming friendly
/// - **Cons**: Lower compression ratio than Deflate/LZMA
/// - **Note**: Focuses on speed over compression ratio, widely used in production systems
///
/// ### Snappy
/// **Use Case**: Very fast compression for Google's production systems
/// - **Best For**: Google BigTable, LevelDB, RocksDB, Apache Kafka, Hadoop
/// - **Time**: O(n) compression (~250 MB/s), O(m) decompression (~500 MB/s)
/// - **Space**: O(hash_table + output)
/// - **Pros**: Extremely fast decompression, simple format, no entropy coding
/// - **Cons**: Lower compression ratio than Deflate (~1.5-2x vs 2-3x)
/// - **Note**: Used in Google's infrastructure, Protocol Buffers compression
///
/// ## Algorithm Selection Guide
///
/// ```
/// Choose RLE when:
/// - Data has long runs of identical values
/// - Speed is critical
/// - Simple implementation needed
///
/// Choose Delta Encoding when:
/// - Working with time series or sequential data
/// - Values change slowly
/// - Planning to use entropy coding afterward
///
/// Choose LZ77 when:
/// - General-purpose compression needed
/// - Data has repeated sequences (not just runs)
/// - Compatible with gzip/PNG/DEFLATE
///
/// Choose LZSS when:
/// - Better compression than LZ77 needed
/// - Embedded systems (simple decode, low memory)
/// - Game assets or ROM compression
/// - Compatible with ARJ/LHA formats
///
/// Choose BWT when:
/// - Maximum compression ratio needed
/// - Can afford preprocessing cost
/// - Planning to use RLE or MTF afterward (like bzip2)
///
/// Choose Huffman Coding when:
/// - Need optimal prefix-free encoding
/// - General-purpose compression (ZIP, GZIP, PNG)
/// - Simple and fast implementation required
/// - Working with standard file formats
///
/// Choose Arithmetic Coding when:
/// - Working with highly skewed probability distributions
/// - Need better compression than Huffman
/// - Building image/video codecs (JPEG 2000, H.264)
/// - Fractional bits per symbol needed
///
/// Choose LZ4 when:
/// - Speed is critical (real-time compression/decompression)
/// - Working with databases or filesystems
/// - Need fast network protocol compression
/// - Moderate compression ratio acceptable
/// - Streaming data or large-scale systems
///
/// Choose Snappy when:
/// - Google-compatible compression needed
/// - Working with BigTable, LevelDB, RocksDB
/// - Apache Kafka or Hadoop integration
/// - Extremely fast decompression priority
/// - Simple format without entropy coding
/// ```
///
/// ## Example: Compression Pipeline
///
/// ```zig
/// const compression = @import("zuda").algorithms.compression;
///
/// // Simple pipeline: BWT → RLE for text
/// var bwt_result = try compression.bwt.encode(allocator, text);
/// defer bwt_result.deinit(allocator);
///
/// var rle_runs = try compression.rle.encode(u8, allocator, bwt_result.data);
/// defer rle_runs.deinit();
///
/// // Decompression: RLE → BWT inverse
/// const rle_decoded = try compression.rle.decode(u8, allocator, rle_runs.items);
/// defer allocator.free(rle_decoded);
///
/// const original = try compression.bwt.decodeFast(allocator, rle_decoded, bwt_result.index);
/// defer allocator.free(original);
/// ```
///
/// ## Compression Metrics
///
/// - **Compression Ratio**: original_size / compressed_size
///   - Ratio > 1: compression achieved
///   - Ratio < 1: data expanded (worse than original)
///   - Ratio = 1: no change
///
/// - **Compressibility**: Varies by algorithm
///   - RLE: Run count vs. data length
///   - Delta: Average delta magnitude
///   - BWT: Character clustering metric
///
/// ## Performance Notes
///
/// - **RLE**: O(n) - fastest, suitable for streaming
/// - **Delta**: O(n) - fast, minimal overhead
/// - **LZ77**: O(n × w) - moderate, adjustable window size
/// - **LZSS**: O(n × w) encode, O(m) decode - better than LZ77, fast decode
/// - **BWT**: O(n² log n) - slowest, use for offline compression
/// - **Huffman**: O(n log k) encode, O(m) decode - fast, widely used
/// - **Arithmetic**: O(n × k) - moderate, k = alphabet size (256 for bytes)
/// - **LZ4**: O(n) encode/decode - fastest, production-ready for real-time systems
/// - **Snappy**: O(n) encode (~250 MB/s), O(m) decode (~500 MB/s) - very fast Google compression
///
/// ## Memory Usage
///
/// - **RLE**: O(k) where k = runs (best case: O(1) for uniform data)
/// - **Delta**: O(n) - same as input
/// - **LZ77**: O(n) output + O(w) window
/// - **LZSS**: O(w) window, typically 40-60% output for text
/// - **BWT**: O(n²) naive, O(n) with optimized implementation
/// - **Huffman**: O(k) tree + codebook where k = alphabet size
/// - **Arithmetic**: O(k) frequency table + O(n) output (k = 256 for bytes)
/// - **LZ4**: O(hash_table) + O(output) - moderate memory, fast processing
/// - **Snappy**: O(hash_table) + O(output) - similar to LZ4, simple format

pub const rle = @import("compression/rle.zig");
pub const delta = @import("compression/delta.zig");
pub const lz77 = @import("compression/lz77.zig");
pub const lzss = @import("compression/lzss.zig");
pub const bwt = @import("compression/bwt.zig");
pub const huffman = @import("compression/huffman.zig");
pub const arithmetic = @import("compression/arithmetic.zig");
pub const lz4 = @import("compression/lz4.zig");
pub const snappy = @import("compression/snappy.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
