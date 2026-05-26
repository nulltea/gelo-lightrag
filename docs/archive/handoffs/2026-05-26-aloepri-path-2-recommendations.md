---
type: handoff
status: stale
created: 2026-05-26
updated: 2026-05-26
tags: [aloepri, path-2, alg2]
companion: [aloepri-attacks]
archive_reason: "Path-2 deployment recommendations extracted from aloepri-attacks.md so the attacks doc stays a pure measurement reference. Recommendations apply to the 2026-05-26 paper-literal Alg2 migration decision."
---

# Handoff — AloePri path-2 deployment recommendations (2026-05-26)

> Extracted from `docs/research/aloepri-attacks.md` §"Implications for path-2".
> The attacks doc retains pure measurement content; this handoff captures the
> deployment decisions that flowed from those measurements.

## Recommendations

1. **The recommended deployment construction is paper-literal Alg2, not our prior default.** Our deployed cell was understating AloePri's actual defense by 7–40 pp on both surfaces. Migration to `--alg2-paper-literal` is the path-2 recommendation, contingent on accuracy preservation under bf16 (paper-literal Û_vo has 500× higher condition number; bf16 inverse loss is a new precision risk to verify — see next-steps memo).
2. **AloePri §5.4 protects the attention output surface, more than we previously measured.** Subject to confirming accuracy under paper-literal, the §5.4-bounded surface defense delta at L=0 is **50 pp** under paper-literal (vs 14 pp under our default). At L≥5 the delta is **40 pp** under paper-literal (vs 0.5 pp under default). This is a substantive deployment protection, not the 1.4 pp we previously reported.
3. **AloePri's score-surface defense, under paper-literal, is also non-trivial at L≥5.** Even outside §5.4's quantitative bound, the paper-literal `kq` defense delta at L=5+ is 16–31 pp, dropping obf to single digits. The L=0 surplus (~5 pp) is still small but no longer "no defense."
4. **A different threat-model reading.** The path-2 score-surface attack we previously characterised as "AloePri provides ~0 pp defense" was a measurement of the *anti-defense version* of Alg2 we'd deployed. Real AloePri Alg2 (paper-literal) defends meaningfully on this surface too. The remaining 6-7 % obf TTRSR at L≥5 is the operational leak budget, not 47 %.
5. **TEE-protected attention (path-1) remains the gold standard for adversaries who can capture either surface at L=0** — even paper-literal Alg2 leaks 43 % on `kq` at L=0 and 47 % on `kqv_out` at L=0. The L=0 surplus is α_e=1.0 embedding-noise shadow; only an in-TEE first decoder layer eliminates it.

## See also

- `docs/research/aloepri-attacks.md` — empirical attack measurements (the data that drives these recommendations)
- `docs/research/aloepri-vs-gelo.md` — applicability of AloePri primitives under GELO openweight threat model
