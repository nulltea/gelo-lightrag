# Attack families in `evals/aloepri-attacks/`

This harness now hosts two attack-research efforts. They share the
loader (`snapshots_loader.py`), the `AttackResult` schema
(`attack_drivers/common.py`), and the memory pre-flight helper
(`m2_7/m2_7_common.py`), but target different threat models and use
different snapshot formats.

| family | tree | threat model | applies to GELO? |
|---|---|---|---|
| **Activation-mask** | `attack_drivers/run_*.py` + `run_all.py` | PCIe attacker sees `U = A·H` for per-forward orthogonal `A` (Haar or HD₃) and an optional shield-stacked operand. Recovers H or H-related structure. | **Yes** — round-3 B.3 gate. C0/C1/C2/C3 conditions. |
| **Static-weight obfuscation** | `m2_7/run_static_attacks.py`, `m2_7/run_ima_embedrow_attacks.py` | Offline attacker has both plain and obfuscated GGUF weights (θ, θ̃). Recovers the τ-permutation that maps plain → obfuscated token ids. | **No** — GELO does not obfuscate the embedding table or any weights. Weights are clear; the protocol's secret is the per-forward mask `A`. Listed here for the AloePri M2.7 GGUF-obfuscation research; not in the GELO acceptance gate. |
| **Token-stream** | `m2_7/run_token_attacks.py` (TFMA, SDA) | Attacker has a stream of obfuscated token ids on the wire. Recovers the τ-permutation from bigram statistics. | **No** — GELO never puts token ids on the wire. The embedding lookup is TEE-internal. Our `attack_drivers/run_tfma.py` and `run_sda.py` are intentionally stubs that emit `not_applicable` rows for this reason. |
| **Hidden-state activation** | `m2_7/run_hidden_state_attacks.py` | Attacker dumps hidden states from the obfuscated llama-server endpoint and attacks them with NN / IMA / IMA-paper-like / ISA. | **Partial overlap** — our `attack_drivers/run_nn.py`, `run_ima.py`, `run_isa.py`, `run_ima_paper_like.py` are independent ports of the same attacks against the GELO PCIe-snapshot format. The M2.7 versions consume llama-server dumps; we consume `InProcessTrustedExecutor` captures. The attack math is the same; only the capture path differs. |
| **Attention-score** | `m2_7/run_hidden_state_attacks.py::_isa_attn_score`, ported to `attack_drivers/run_isa_attn_score.py` | Attacker sees per-head attention scores `(n_heads, n_q, n_kv)` and runs ridge ISA. | **Deferred** — GELO keeps attention compute in-TEE per the M1.3 design lock; attention scores never cross PCIe. Becomes applicable when the M1.10 fused-permuted-attention path moves attention to the GPU under the permuted protocol. The driver is in the harness and emits `not_applicable` until the protocol adds a `WeightKind::AttnScore` snapshot kind. |

## Why static-weight attacks don't apply to GELO

The static-weight attack family (path-2 / paper §F.1) targets the
**obfuscated weights** themselves: `W̃_embed[i] = W_embed[τ(i)]` for
some permutation τ. The attacker knows the plain weights θ and the
obfuscated weights θ̃ (the GGUF being shipped) and tries to recover τ
— which would let them decode every wire-side prompt to that
deployment.

GELO does not ship obfuscated weights. The mask `A` is sampled
per-forward inside the TEE and is never embedded in any deployed
artefact. There is no static τ for the static-weight attacks to
recover, so the family is mathematically inapplicable to the GELO
threat model.

If a *future* GELO variant ships weight-level obfuscation alongside
per-forward masking (a "double-defence" mode), the M2.7 drivers
would become directly relevant. Until then they live in `m2_7/`
as documentation of a parallel attack-research effort.

## Corpora

* `corpora/release-gate-64.txt` — 64-prompt hand-curated set.
  Sufficient for AloePri-family control thresholds
  (`c0_ima_at_least_95pct`, `c0_ima_paper_like_at_least_50pct`).
* `corpora/release-gate-512.txt` — 512-prompt set (curated 64 +
  448 filtered PIQA goal+solution sentences). Use for release-gate
  runs where you need tight statistics on `nn` / `anchor_ica` /
  `jade` / `jd`.
* The Rust capture binary's built-in `SMOKE_PROMPTS` (8 prompts)
  is a smoke fixture only — clears the pipeline but leaves
  inversion-model trainers under-sampled, which is why the c0
  controls fail on the smoke corpus.

## Memory pre-flight

The `m2_7/m2_7_common.py::check_phase_memory(phase)` helper is
imported by `run_all.py` and fires at startup with `phase=
"attack_matrix"` (8 GB minimum). The Rust capture binary has its
own `--min-mem-gb` flag (8 GB default; covers Qwen3-1.7B f32
weights + engine upload buffers + safetensors export staging).
Both honour `--skip-mem-check` for operators who've measured
headroom.
