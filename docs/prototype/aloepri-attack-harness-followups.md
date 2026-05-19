# AloePri attack harness — deferred attacks + follow-ups

Companion to:
- [`aloepri-attack-harness.md`](aloepri-attack-harness.md) — the Phase 2 spec.
- [`aloepri-attack-harness-findings.md`](aloepri-attack-harness-findings.md) — the OOM incident + safeguards.

This file tracks attacks the harness either **does not yet implement** or that depend on Phase 1 capture-surface extensions still to be built. It is the source of truth for "what's not measured" — referenced from `run_all.py`'s acceptance-gate documentation.

## D1 — Cross-batch Gram-leak attack (defends what the shield was designed for)

**Status:** deferred. Was placeholder gate `c1_strictly_higher_than_c2_on_isa`, replaced 2026-05-19 with `per_offload_at_most_per_forward_plus_shield_on_isa` (correct trade-off; *different threat*).

**Attack:** the PCIe attacker collects masked operands `U_b = A_b · stack(H_b, S_b)` across many forward passes and builds the empirical Gram matrix `Σ_b Uᵀ_b U_b`. Under per-batch fresh Haar mask the off-diagonal information cancels in expectation, but the on-diagonal terms collapse to `Σ_b (HᵀH + SᵀS)` (per `docs/prototype/gelo.md` §3.3, TwinShield). **Without shield rows** the attacker recovers `Σ_b HᵀH` directly — a row-level token-frequency leak across the entire request stream. **With shield rows** the `SᵀS` term injects calibrated noise into the Gram, masking the `HᵀH` summary statistic.

**Why it matters:** this is the exact threat the shield primitive (`ShieldConfig::new(8, 4.0)` C2 default) was designed to defend. None of our current attacks (NN, IMA, ISA, IMA paper-like) probe this — they all operate within a single forward pass, while the shield's defence is **across forward passes**.

**Implementation sketch:**

1. Capture snapshots across many prompts in **one condition** (already done by `capture_snapshots`).
2. Build `Σ_b Uᵀ_b U_b` from all per-prompt operands at a fixed (layer, kind).
3. Attack 1: ASR against recovering token frequencies. Compute spectrum / top-eigenvector overlap with the plaintext embedding-table singular vectors. If the shield works, this overlap should be low.
4. Attack 2: NN-style matching of Gram rows against the embedding table (extended NN). Threshold for "shield works" is similar overlap drop.

**Why we deferred:** the attack needs a different data layout than what the current Python harness consumes (per-prompt rows, not aggregated Gram matrices), and the meaningful comparison requires C1-with-shield-on as a fourth condition (currently `with_per_offload_mask()` *clears* the shield in `sim.rs:286`). That's a protocol-side change to allow `with_per_offload_mask().with_shield_on()`, which we held off on per "do not change any gelo related code".

**Acceptance criterion when implemented:** `gram_leak_c2_below_5pct` (a new gate). Threshold informed by AloePri Table 1 baseline.

## D2 — Attention-score (`Q · Kᵀ`) observable capture

**Status:** deferred. Phase 1 snapshot module hooks `offload_linear`, `offload_qkv`, `offload_linear_many`; the attention-score path (`offload_attention_qkt`, `offload_attention_permuted_cached`) is **not** hooked. Per AloePri Table 3 (§6.5 ablation) the attention-score surface is the harder one — ISA scores 87.14% on AttnScore vs 40% on HiddenState under "Noise only", and only Head&BlockPerm drops AttnScore to 0%.

**Implementation cost:** add `SnapshotCapture::record_attention_score(...)` hook + plumbing through `offload_attention_qkt_batched` / `offload_attention_permuted{,_cached}` in `gelo-protocol`. Touches the API contract we explicitly froze at Phase 1 boundary, so requires a coordinated re-freeze.

## D3 — KV-cache observable

**Status:** deferred. AloePri's `matrix.py` lists `kv_cache` as a separate ISA observable type. We don't currently capture KV-cache contents — the KV cache stays inside `KvCache`, never crosses the executor's offload seam.

**Threat model relevance:** if the GPU has VRAM read access (our threat model per `gelo.md` §2), the KV cache **is** observable. The current harness doesn't probe it because the snapshot module operates at the `offload_*` API surface, not at the VRAM-resident KV layout. A separate capture-after-write hook on `KvCache::append` would unblock this.

## D4 — Full 256-prompt corpus for IMA paper-like

**Status:** known limitation. The paper-like 2-layer inverter undertrains at 64 prompts (~680 training rows). AloePri's `IMAPaperLikeConfig` defaults to `train_sequence_count=128` × `sequence_length=32` = 4096 rows. To match that scale we need ~256 prompts × ~16 tokens.

**Effect on gates:** `c0_ima_paper_like_at_least_50pct` will fail the fast variant (8–64 prompts) and only meaningfully pass on the release-gate full corpus. The CI fast-variant gate should either:
1. Skip this check for `--max-prompts < 200`, or
2. Use `c0_ima_at_least_95pct` (ridge IMA) as the C0 sanity instead — that one DOES work at 64 prompts (now 96.4% with multi-alpha selection).

## D5 — Sensitive-token filter (PIIRSR)

**Status:** not ported. AloePri's reference defines `_collect_sensitive_plain_ids` and reports a separate `PIIRSR` (PII Recovery Success Ratio) on the sensitive-token slice. Useful for medical-domain or financial-domain deployments. Not in our gate yet.

**Implementation cost:** small — wrap the existing IMA/ISA evaluation to filter test_ids to a PII subset, recompute TTRSR on the slice. AloePri's PII list (`vendor/aloepri-py/src/security_qwen/ima.py::_collect_sensitive_plain_ids`) is reusable.

## D6 — Real VMA (RowSort weight-pair attack)

**Status:** structurally inapplicable. AloePri's VMA recovers Π from `(W_plain, W_obfuscated)` pairs. GELO doesn't obfuscate weights → no `W_obfuscated` → no Π to recover. The driver `run_vma.py` is a `not_applicable` stub documenting this. **No follow-up planned** unless GELO's threat model adds a weight-obfuscation defence (it won't — openweight is the design point).

---

## Tracking

| ID | Attack / surface | Driver | Snapshot capture | Gate |
|---|---|---|---|---|
| D1 | Cross-batch Gram leak | not built | already captured | not_yet (placeholder removed) |
| D2 | AttnScore observable | not built | **not** captured | n/a |
| D3 | KV-cache observable | not built | **not** captured | n/a |
| D4 | IMA paper-like at scale | built (`run_ima_paper_like`) | already captured | needs ≥ 200 prompts |
| D5 | PIIRSR slice | not built | already captured | not_yet |
| D6 | RowSort weight-pair VMA | n/a — inapplicable | n/a | n/a (stub) |
