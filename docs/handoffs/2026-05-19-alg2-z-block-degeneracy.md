# Finding — `Ẑ_block` is silently identity at alg2.py defaults

**Date:** 2026-05-19
**Discovered:** while validating the matrix-Γ kernel patch (Option C
of `2026-05-19-alg2-qwen3-shape-analysis.md`).
**Status:** documented; doesn't block Option C; worth fixing in a
follow-up if paper Table 3's "Head&BlockPerm" headline number matters
for the deployment narrative.

## What the bug is

`python/aloepri-llm/lib/alg2.py:generate_block_perm` and the equivalent in
`vendor/aloepri-py/src/attention_keys.py` produce the **identity
permutation matrix** for every seed across head_dim ∈ {8, 16, 32, 64,
128} under the **default parameters** (`beta=8`, `gamma=1e3`,
`rope_base=1e6`).

The function samples a "window size" per step from a softmax over
ζ-log differences:

```python
zeta_log[idx] = (-2 * idx / num_blocks) * log(rope_base)
local_scores  = gamma * (zeta_log[start:start+c] - zeta_log[start])
probs         = softmax(local_scores)
window_size   = rng.choice(c, p=probs) + 1
```

With `rope_base = 1e6`, `log(rope_base) ≈ 13.8`. The `zeta_log`
differences for `start=0`, `c=beta=8`, `num_blocks=64` (i.e. Qwen3-1.7B's
`d_h/2`) are roughly `[0, -0.43, -0.87, …, -3.0]` — already in the
negative tens after multiplication by `gamma=1e3`. The softmax collapses
to `[1.0, ~0, ~0, …]` to fp32 precision; `rng.choice` always returns
index 0; window size is always 1 → no shuffle ever happens → returned
permutation is the identity.

I sweep-verified this across seeds {7, 42, 99, 12345} and head_dim ∈
{8, 16, 32, 64, 128}: `z² - I = 0` and `z is identity? True` in every
case.

## Why it didn't surface earlier

The §05 Qwen3 deployment forces `q_matrix = k_matrix = I` (see
`obfuscate_qwen3_gguf.py:362-368`), so the intra-head transforms — and
therefore `z_block` — were not exercised in inference. M2.7 attack
ledger numbers reflect head-shuffle + keymat + α-noise only; the
"block permutation" column of paper Table 3 was never actually wired
through to runtime on path-2.

## What paper Table 3 actually attributes to "Head&BlockPerm"

Table 3 lumps head permutation (`Π_head`, paper §5.2.4) and block
permutation (`Ẑ_block`, paper §5.2.3) together under "Head&BlockPerm".
"Noise + KeyMat" → 0.82 % HS, "Noise + KeyMat + Head&BlockPerm" → 0 %.
The marginal contribution of the BlockPerm half alone is not isolated.
It could be small (most of the win comes from Π_head); it could be
large.

## Impact on the Option C deployment

`Ẑ_block` being identity does **not** break the matrix-Γ kernel
algebra. With `z = I` and `qk_scale_range = (1.0, 1.0)` (the MVP
choice forced by Option C for orthogonality), the per-layer M_q
reduces to `R̂_qk` — the RoPE-aware rotation per NEOX pair, alone.
The matrix-Γ algebra is still exact under that M_q.

What it *does* affect: the strength of the intra-head obfuscation
that the matrix-Γ kernel ferries through. We deploy R̂_qk + (head
shuffle from `Π_head` via `tau_kv`/`tau_group`). We don't deploy
Ĥ_qk (dropped by the orthogonality MVP) or Ẑ_block (silently I).

So the path-2 Option C deployment is **R̂_qk + Π_head only** out of
the paper's three named intra-head primitives. If M2.7 IMA still
crosses the 15 % gate after Option C, this is the obvious next
component to investigate fixing.

## Fix sketches (for a follow-up)

The softmax flattens because `gamma · ΔζLog` dominates. Options:

- **(a)** Drop `gamma` to ~1 or below. Then `ΔζLog ≈ -0.43` puts
  meaningful probability on adjacent indices, window size > 1.
- **(b)** Use `rope_base` ~ `1e3` in the score function only (not the
  model's actual RoPE base). Then `log(rope_base) ≈ 7` and ΔζLog is
  smaller, softmax less peaked.
- **(c)** Replace the dynamic-window softmax with a uniform random
  permutation over the full `num_blocks` (no windowing). Cleanest;
  matches paper §5.2.3 description more literally.

Each requires a re-derivation of why the paper chose the
dynamic-window form — there's likely a RoPE-frequency-locality
argument we shouldn't break. Defer until M2.7 re-run with Option C
tells us whether `Ẑ_block` is load-bearing.

## Verification recipe

```bash
cd python/aloepri-llm
.venv/bin/python -c "
from lib import alg2
import numpy as np
for d in (8, 16, 32, 64, 128):
    for s in (7, 42, 99, 12345):
        z = alg2.generate_block_perm(
            num_blocks=d//2, beta=8, gamma=1e3,
            rope_base=1e6, seed=s,
        )
        assert np.allclose(z, np.eye(d)), f'd={d} s={s} non-identity'
print('all identity')
"
```
