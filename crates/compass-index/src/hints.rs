//! Directional Neighbor Filtering (Compass §4.5).
//!
//! For each layer-0 node `x` and each of its `M` neighbours `n_i`, we
//! precompute a quantised direction vector pointing from `x` toward
//! `n_i`. The hints are cached cleartext inside the CVM; their full
//! size is `num_nodes · M · D · directional_hint_bits / 8` bytes,
//! orders of magnitude smaller than the embedding dataset.
//!
//! At search time, when the beam-search visits a candidate `c`:
//!   - The candidate's *full* embedding is already in hand (it was
//!     ORAM-fetched in the prior round).
//!   - We compute `q_dir = q - emb(c)` — the direction from `c` toward
//!     the query.
//!   - For each of `c`'s `M` neighbours we score
//!     `dot(dequant(hint), q_dir)`; the higher the score, the more
//!     likely that neighbour lies in the query's direction relative
//!     to `c`.
//!   - We ORAM-fetch only the top `ef_n` neighbours (`ef_n < M`).
//!     The remaining `M - ef_n` neighbours stay in the tree, unread.
//!
//! This is the headline Compass optimisation. Combined with §4.6
//! Speculative Neighbor Prefetch (M4.5) and §4.7 Lazy Eviction
//! (M4.1, shipped), the paper reports up to 920× speedup over a
//! strawman HNSW-over-ORAM.

/// Quantisation grid for a 4-bit signed component: 16 levels mapping
/// `[-1, 1]` to `[0, 15]`. Adequate at typical embedding ranges
/// (unit-norm vectors).
const Q_LEVELS: u8 = 15;

/// Quantise one f32 in `[-1, 1]` to a 4-bit code in `[0, 15]`. Values
/// outside the range are clamped; this loses a constant offset for
/// non-unit embeddings but preserves direction.
pub(crate) fn quantise_nibble(x: f32) -> u8 {
    let clamped = x.clamp(-1.0, 1.0);
    let mapped = (clamped + 1.0) / 2.0 * Q_LEVELS as f32;
    mapped.round() as u8 & 0x0f
}

/// Dequantise a 4-bit code back to `[-1, 1]`.
pub(crate) fn dequantise_nibble(q: u8) -> f32 {
    (q as f32) / (Q_LEVELS as f32) * 2.0 - 1.0
}

/// Pack a D-dim f32 vector into `ceil(D/2)` bytes of 4-bit codes.
/// The vector is first normalised by its L∞ norm to bring it into
/// `[-1, 1]` (preserves direction; loses scale, which is irrelevant
/// for cosine-like scoring).
pub(crate) fn pack_direction(direction: &[f32]) -> Vec<u8> {
    let scale = direction
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f32, f32::max)
        .max(f32::MIN_POSITIVE);
    let normed: Vec<f32> = direction.iter().map(|x| x / scale).collect();

    let mut out = vec![0u8; (normed.len() + 1) / 2];
    for (i, &x) in normed.iter().enumerate() {
        let q = quantise_nibble(x);
        if i % 2 == 0 {
            out[i / 2] |= q;
        } else {
            out[i / 2] |= q << 4;
        }
    }
    out
}

/// Unpack a quantised hint into f32. Reciprocal of [`pack_direction`].
/// The scale that was applied at pack time is *lost*; the returned
/// vector is in `[-1, 1]` and carries direction only.
#[cfg(test)]
pub(crate) fn unpack_direction(packed: &[u8], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        let nibble = if i % 2 == 0 {
            packed[i / 2] & 0x0f
        } else {
            packed[i / 2] >> 4
        };
        out.push(dequantise_nibble(nibble));
    }
    out
}

/// Dot product of a query direction and a hint, computed without
/// materialising the unpacked hint as a `Vec`. Hot path during
/// search.
pub(crate) fn score_packed(packed: &[u8], q_dir: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (i, &qd) in q_dir.iter().enumerate() {
        let nibble = if i % 2 == 0 {
            packed[i / 2] & 0x0f
        } else {
            packed[i / 2] >> 4
        };
        acc += dequantise_nibble(nibble) * qd;
    }
    acc
}

/// Hints for one layer-0 node — one packed direction per neighbour,
/// stored in the same order as `NodeBlock::neighbors`.
#[derive(Debug, Clone)]
pub(crate) struct NodeHints {
    /// `packed_hints[i]` is the quantised direction from this node
    /// toward its `i`-th neighbour. Each packed vector is
    /// `ceil(D/2)` bytes.
    pub(crate) packed_hints: Vec<Vec<u8>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantise_dequantise_round_trip_within_tolerance() {
        for &x in &[-1.0_f32, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0] {
            let q = quantise_nibble(x);
            let back = dequantise_nibble(q);
            // 4-bit grid: 16 levels, step ≈ 0.133 ⇒ max error ≈ 0.067.
            assert!((back - x).abs() < 0.08, "x={x} back={back}");
        }
    }

    #[test]
    fn pack_unpack_preserves_direction_sign() {
        let dir = vec![0.8, -0.6, 0.2, -0.9, 0.0, 0.4];
        let packed = pack_direction(&dir);
        let unpacked = unpack_direction(&packed, dir.len());
        // Sign agreement on every non-zero component.
        for (a, b) in dir.iter().zip(unpacked.iter()) {
            if a.abs() > 0.05 {
                assert_eq!(
                    a.signum(),
                    b.signum(),
                    "sign mismatch: a={a} b={b}"
                );
            }
        }
    }

    #[test]
    fn score_packed_matches_unpacked_dot() {
        let dir = vec![0.7, -0.4, 0.1, 0.9, -0.2, 0.3];
        let packed = pack_direction(&dir);
        let unpacked = unpack_direction(&packed, dir.len());
        let q = vec![0.5, -0.5, 0.2, 0.8, 0.0, -0.3];
        let direct_dot: f32 = unpacked.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        let scored = score_packed(&packed, &q);
        assert!((direct_dot - scored).abs() < 1e-5, "{direct_dot} vs {scored}");
    }

    #[test]
    fn score_packed_ranks_aligned_higher_than_orthogonal() {
        let neighbour_dir = vec![1.0, 0.0, 0.0, 0.0];
        let packed = pack_direction(&neighbour_dir);
        let query_aligned = vec![0.9, 0.1, 0.0, 0.0];
        let query_orthogonal = vec![0.0, 1.0, 0.0, 0.0];
        let score_aligned = score_packed(&packed, &query_aligned);
        let score_orthogonal = score_packed(&packed, &query_orthogonal);
        assert!(
            score_aligned > score_orthogonal,
            "aligned {score_aligned} <= orthogonal {score_orthogonal}"
        );
    }
}
