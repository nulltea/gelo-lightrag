//! Tree-layout math. Buckets are numbered breadth-first: root = 0,
//! children of bucket `b` are `2b + 1` and `2b + 2`. Leaves form the
//! contiguous range `[2^(L-1) - 1 .. 2^L - 1)`.
//!
//! A `PathId p ∈ [0, 2^(L-1))` selects leaf `2^(L-1) - 1 + p`. The
//! corresponding `path_buckets(p)` traverses root → that leaf.

/// 0-based leaf index. `PathId(p)` ⇔ leaf bucket `2^(L-1) - 1 + p`.
/// 32 bits is enough: 2^31 leaves ⇒ ~10⁹ real blocks at `Z = 4`, well
/// beyond the LightRAG enterprise-scale ceiling (10⁶ entities).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PathId(pub u32);

/// Number of tree levels needed to host `n_leaves` distinct paths.
/// `n_leaves` need not be a power of two; we round up so every path
/// has a leaf. `n_leaves == 1` ⇒ 1 level (the root *is* the leaf).
pub fn tree_levels(n_leaves: u32) -> u32 {
    if n_leaves <= 1 {
        return 1;
    }
    32 - (n_leaves - 1).leading_zeros() + 1
}

/// Total bucket count for a tree with `n_leaves` paths.
pub fn total_buckets(n_leaves: u32) -> u32 {
    (1u32 << tree_levels(n_leaves)) - 1
}

/// Bucket indices on the path from root to leaf `p`, in root-first
/// order. Length is exactly `tree_levels(n_leaves)`.
pub fn path_buckets(p: PathId, n_leaves: u32) -> Vec<u32> {
    let levels = tree_levels(n_leaves);
    let leaf_bucket = (1u32 << (levels - 1)) - 1 + p.0;
    let mut out = Vec::with_capacity(levels as usize);
    let mut cur = leaf_bucket;
    out.push(cur);
    for _ in 1..levels {
        cur = (cur - 1) / 2;
        out.push(cur);
    }
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_round_up() {
        assert_eq!(tree_levels(1), 1);
        assert_eq!(tree_levels(2), 2);
        assert_eq!(tree_levels(4), 3);
        assert_eq!(tree_levels(5), 4); // 5 leaves needs 8-leaf tree (4 levels)
        assert_eq!(tree_levels(64), 7);
    }

    #[test]
    fn total_buckets_matches_complete_tree() {
        assert_eq!(total_buckets(1), 1);
        assert_eq!(total_buckets(2), 3);
        assert_eq!(total_buckets(4), 7);
        assert_eq!(total_buckets(64), 127);
    }

    #[test]
    fn path_for_first_leaf_of_4_leaf_tree() {
        // Levels = 3, leaf bucket for path 0 is bucket 3.
        // Parent chain: 3 → 1 → 0.
        let path = path_buckets(PathId(0), 4);
        assert_eq!(path, vec![0, 1, 3]);
    }

    #[test]
    fn path_for_last_leaf_of_4_leaf_tree() {
        // Levels = 3, leaf bucket for path 3 is bucket 6.
        // Parent chain: 6 → 2 → 0.
        let path = path_buckets(PathId(3), 4);
        assert_eq!(path, vec![0, 2, 6]);
    }

    #[test]
    fn paths_share_root_only() {
        let p0 = path_buckets(PathId(0), 8);
        let p7 = path_buckets(PathId(7), 8);
        assert_eq!(p0[0], 0);
        assert_eq!(p7[0], 0);
        // Any further down they diverge.
        assert_ne!(p0[1], p7[1]);
    }

    #[test]
    fn path_length_equals_levels() {
        for &n in &[1u32, 2, 4, 8, 16, 64, 128, 1024] {
            let levels = tree_levels(n) as usize;
            for p in [0, n / 2, n - 1] {
                let path = path_buckets(PathId(p), n);
                assert_eq!(path.len(), levels, "n_leaves={n} path={p}");
            }
        }
    }
}
