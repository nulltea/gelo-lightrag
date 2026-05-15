//! In-TEE scoring helpers — sort, top-k, tie-shuffle.
//!
//! All operations run on the trusted side: scores never escape. The
//! ranked output feeds [`crate::output::EncryptedRerankBundle::seal`],
//! which is the only path through which any rerank information crosses
//! the trust boundary.

use rag_core::ChunkId;
use rand::seq::SliceRandom;
use rand::Rng;

/// Scored candidate before sorting.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub chunk_id: ChunkId,
    pub text: String,
    pub score: f32,
}

/// Sorted candidate after `top_k_with_tie_shuffle`.
#[derive(Debug, Clone)]
pub struct RankedItem {
    pub rank: u32,
    pub chunk_id: ChunkId,
    pub text: String,
}

/// Sort `scored` by score descending, break ties with a session-keyed
/// shuffle on equal-score buckets, return the first `top_k`. The
/// shuffle uses `rng` (typically a `ChaCha20Rng` seeded from the query
/// key) so tied scores don't reveal stable secondary order to a host
/// observer who can see the rank-by-rank emission order.
///
/// `score = f32::NAN` is treated as the worst possible score (NaNs go
/// to the back).
pub fn top_k_with_tie_shuffle<R: Rng + ?Sized>(
    mut scored: Vec<ScoredCandidate>,
    top_k: usize,
    rng: &mut R,
) -> Vec<RankedItem> {
    if scored.is_empty() || top_k == 0 {
        return Vec::new();
    }

    scored.sort_by(|a, b| {
        match (a.score.is_nan(), b.score.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    // Tie-shuffle: find runs of equal score and shuffle within each run.
    let mut i = 0;
    while i < scored.len() {
        let mut j = i + 1;
        while j < scored.len() && scored[j].score == scored[i].score {
            j += 1;
        }
        if j - i > 1 {
            scored[i..j].shuffle(rng);
        }
        i = j;
    }

    scored
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(idx, c)| RankedItem {
            rank: idx as u32,
            chunk_id: c.chunk_id,
            text: c.text,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn cand(id: &str, text: &str, score: f32) -> ScoredCandidate {
        ScoredCandidate {
            chunk_id: ChunkId(id.into()),
            text: text.into(),
            score,
        }
    }

    #[test]
    fn sort_picks_highest_score_first() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let out = top_k_with_tie_shuffle(
            vec![
                cand("a", "alpha", 0.3),
                cand("b", "beta", 0.9),
                cand("c", "gamma", 0.6),
            ],
            3,
            &mut rng,
        );
        assert_eq!(out[0].chunk_id.0, "b");
        assert_eq!(out[1].chunk_id.0, "c");
        assert_eq!(out[2].chunk_id.0, "a");
        assert_eq!(out[0].rank, 0);
        assert_eq!(out[1].rank, 1);
        assert_eq!(out[2].rank, 2);
    }

    #[test]
    fn top_k_truncates() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let out = top_k_with_tie_shuffle(
            vec![
                cand("a", "alpha", 0.1),
                cand("b", "beta", 0.2),
                cand("c", "gamma", 0.3),
                cand("d", "delta", 0.4),
                cand("e", "epsilon", 0.5),
            ],
            2,
            &mut rng,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].chunk_id.0, "e");
        assert_eq!(out[1].chunk_id.0, "d");
    }

    #[test]
    fn tied_scores_shuffle_within_bucket() {
        // Three tied at 0.5 — over many seeds we expect different orderings.
        let mut orderings = std::collections::HashSet::new();
        for seed in 0..32u64 {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let out = top_k_with_tie_shuffle(
                vec![
                    cand("a", "alpha", 0.5),
                    cand("b", "beta", 0.5),
                    cand("c", "gamma", 0.5),
                ],
                3,
                &mut rng,
            );
            orderings.insert(out.iter().map(|r| r.chunk_id.0.clone()).collect::<Vec<_>>());
        }
        // At least 2 distinct orderings across 32 seeds (very high
        // probability if shuffle is uniform; deterministic test below
        // proves single-seed reproducibility).
        assert!(orderings.len() >= 2, "tie shuffle is not random");
    }

    #[test]
    fn tie_shuffle_is_deterministic_given_seed() {
        let make = || {
            vec![
                cand("a", "", 0.5),
                cand("b", "", 0.5),
                cand("c", "", 0.5),
            ]
        };
        let mut r1 = ChaCha20Rng::seed_from_u64(7);
        let mut r2 = ChaCha20Rng::seed_from_u64(7);
        let o1 = top_k_with_tie_shuffle(make(), 3, &mut r1);
        let o2 = top_k_with_tie_shuffle(make(), 3, &mut r2);
        assert_eq!(
            o1.iter().map(|r| r.chunk_id.0.clone()).collect::<Vec<_>>(),
            o2.iter().map(|r| r.chunk_id.0.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn nan_scores_go_last() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let out = top_k_with_tie_shuffle(
            vec![
                cand("a", "alpha", f32::NAN),
                cand("b", "beta", 0.1),
                cand("c", "gamma", 0.2),
            ],
            3,
            &mut rng,
        );
        assert_eq!(out[0].chunk_id.0, "c");
        assert_eq!(out[1].chunk_id.0, "b");
        assert_eq!(out[2].chunk_id.0, "a");
    }
}
