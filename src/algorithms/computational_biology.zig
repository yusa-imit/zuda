/// Computational Biology Algorithms
///
/// This module provides algorithms for analyzing biological sequences (DNA, RNA, proteins),
/// including sequence alignment, pattern matching, and molecular biology operations.
///
/// **Categories**:
/// - **Sequence Alignment**: Global (Needleman-Wunsch) and local (Smith-Waterman) alignment
/// - **Pattern Matching**: KMP-based motif finding in biological sequences
/// - **Molecular Biology**: Transcription, translation, reverse complement, GC content
///
/// **Use Cases**:
/// - Bioinformatics: sequence comparison, homology detection, evolutionary analysis
/// - Genomics: gene finding, motif discovery, restriction site mapping
/// - Proteomics: protein domain identification, functional annotation
///
/// **Algorithms**: 16th category in zuda library

pub const sequence_alignment = @import("computational_biology/sequence_alignment.zig");
pub const pattern_matching = @import("computational_biology/pattern_matching.zig");

// Re-export key types and functions
pub const Alignment = sequence_alignment.Alignment;
pub const ScoreFunction = sequence_alignment.ScoreFunction;
pub const DNA_SCORE = sequence_alignment.DNA_SCORE;
pub const PROTEIN_SCORE = sequence_alignment.PROTEIN_SCORE;

pub const needlemanWunsch = sequence_alignment.needlemanWunsch;
pub const smithWaterman = sequence_alignment.smithWaterman;
pub const findPattern = pattern_matching.findPattern;
pub const reverseComplement = pattern_matching.reverseComplement;
pub const gcContent = pattern_matching.gcContent;
pub const transcribe = pattern_matching.transcribe;
pub const translate = pattern_matching.translate;

test {
    @import("std").testing.refAllDecls(@This());
}
