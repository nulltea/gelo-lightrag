use crate::types::EncryptedEmbedding;

pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

pub fn l2_norm(vector: &[f32]) -> f32 {
    dot(vector, vector).sqrt()
}

pub fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    let lhs_norm = l2_norm(lhs);
    let rhs_norm = l2_norm(rhs);

    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        return 0.0;
    }

    dot(lhs, rhs) / (lhs_norm * rhs_norm)
}

pub fn top_k_by_similarity<'a>(
    query: &[f32],
    vectors: impl IntoIterator<Item = &'a EncryptedEmbedding>,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = vectors
        .into_iter()
        .enumerate()
        .map(|(idx, embedding)| (idx, cosine_similarity(query, &embedding.vector)))
        .collect();

    scored.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::cosine_similarity;

    #[test]
    fn cosine_similarity_matches_expectation() {
        let lhs = [1.0, 0.0, 0.0];
        let rhs = [0.5, 0.5, 0.0];
        let sim = cosine_similarity(&lhs, &rhs);
        assert!(sim > 0.70 && sim < 0.71);
    }
}
