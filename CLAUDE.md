# Repository conventions for Claude

This file captures project-wide rules that apply regardless of which
file or crate is being edited. Loaded into context automatically.

## Gelo-LLM main benchmark

The canonical performance benchmark for the GELO LLM serving path is
the `gelo_llm_prefill_decode_breakdown` test in
`crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`. It loads
real Qwen3 weights, runs one prefill of `n` tokens then `K` decode
steps at batch `B`, and dumps the **full per-op profile** for each
phase — GPU buckets (`engine:matmul` single-weight O/FfnDown/LM-head,
`engine:matmul_many` fused QKV + gate∥up) alongside the in-TEE CPU
buckets (`tee:attn_cached`, `gelo:mask_apply/unapply:*`, shield). It
profiles the production default (R3 LM-head GPU offload) only.

Canonical invocation (production shape; `--release` is mandatory — a
debug build pays minute-scale cubecl shader compile/autotune):

```bash
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 GELO_BENCH_MAX_TOKENS=32 \
  cargo test --release -p gelo-gpu-wgpu \
  --test qwen3_m1_12_r1_q1_microbench \
  gelo_llm_prefill_decode_breakdown -- --ignored --nocapture
```

Env knobs: `GELO_BENCH_VARIANT` (`1.7B` default, `4b`), `GELO_BENCH_B`
(default 8), `GELO_BENCH_N` (default 2048), `GELO_BENCH_MAX_TOKENS`
(default 64). Weights download once to `~/.cache/huggingface`.

Build prerequisites (no apt, no sudo): install the Rust toolchain via
`rustup`, then build AOCL-BLIS once with
`scripts/install-aocl-blis.sh` (the `blas` feature is default-on and
links `vendor/aocl-install/lib/libblis-mt.so`). The wgpu Vulkan engine
auto-selects the best adapter — the discrete GPU on a dGPU box, the
iGPU on Strix Halo. No ROCm/HIP is required.

Results live in the perf chronicles under `docs/dev/logs/`:
`gelo-llm-perf-chronicle.md` (iGPU / Strix Halo) and
`gelo-llm-perf-chronicle_dgpu.md` (dGPU / Nvidia).

## HTML docs (`docs/prototype/*.html`)

These are **design + result documents**, not code documentation and
not a development log.

- **Voice:** describe *what the system does* and *why* (the design
  decisions and their measured outcomes). Not *what the code looks
  like*.
- **Minimise code references.** Cite a path/identifier only when it
  is load-bearing for understanding the design call (e.g. "the
  `MaskKind::Auto` threshold picks HD₃ at pad ratio ≤ 4/3"). Never
  inline a `pub fn …` signature, a struct definition, a use-statement,
  or a bullet list of method names. The HTML is a public artefact;
  function signatures belong in rustdoc / code, not here.
- **No code-as-narrative.** "We added `score_candidates_batched`,
  `run_decode_step_batched`, …" is a development log — it tells the
  reader *what we built* rather than *what the system now does*. Lead
  with the architectural choice ("at decode each sequence contributes
  one row under one shared mask"); cite the code path only as a
  pointer for further reading.
- **Tables and figures over prose-lists.** Use comparative tables
  (before/after, baseline/variant) to present results. Avoid bullet
  lists of "added X, refactored Y, fixed Z".
- **Keep history compact.** If a section accumulates multiple
  optimisation rounds, summarise the cumulative outcome and a short
  note on the lever that made the headline shift. Don't enumerate
  every commit-step.
- **No plan-time / development aliases.** Replace milestone tags and
  in-session labels with self-descriptive names that describe what
  the design *is*, not which sprint it landed in. Examples to avoid:
  `M1.10` / `M1.11` / `M1.12`, `Phase 1a` / `Phase 1b`, `D1.6` /
  `R1.4`, `Path 1` / `Path 2`, `Option A` / `Option B` / `Option C`,
  `v1` / `v7` (when used as release tags). Rewrite as the design
  name: "M1.11 batched stack" → "batched substrate"; "M1.10 fused
  permuted attention" → "fused permuted attention"; "Option A
  (Auto)" → "the Auto-dispatch design"; "v1-demonstrator blocker" →
  "blocker for the current Qwen3 demonstrator path". Plans, handoffs,
  and memory files may keep the alias for cross-reference; the public
  HTML artefact should not — the milestone tag decays, the design
  persists. (Same rule applies to code identifiers per
  `feedback_public_docs_no_aliases.md`.)
- **Formatting:** run `npx prettier --write <file>` after every edit
  (per memory `feedback_prettier_after_html_edits.md`). HTML edits
  that skip prettier produce hard-to-review noise in subsequent
  diffs.

When in doubt: would this paragraph make sense to a reader who has
never seen the source tree? If it depends on knowing the crate
layout or having `cargo doc` open, it belongs in code/rustdoc, not
the HTML.

## Markdown docs (`docs/**/*.md`)

All markdown docs under `docs/` carry YAML frontmatter:

```yaml
---
type: <handoff|plan|prototype-note|research|theory|dev-log|reference>
status: <current|partial|stale>
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: []
# Optional: superseded_by, supersedes, companion, archive_reason
---
```

Folder mapping (by `type`):

- `handoff`        → `docs/handoffs/YYYY-MM-DD-<slug>.md` (filename date = last update)
- `plan`           → `docs/plans/`
- `prototype-note` → `docs/dev/prototype/`
- `research`       → `docs/research/`
- `theory`         → `docs/research/`
- `dev-log`        → `docs/dev/logs/`
- `reference`      → `docs/plans/` (no dedicated folder until critical mass)

When a handoff is no longer in active reference (typically more than a
few days old and not driving current work), move it to
`docs/archive/handoffs/`. The filename stays the same. The active
`docs/handoffs/` directory should hold only handoffs that the current
or next session is likely to read. Plans and other docs that go stale
(superseded, aborted, paused indefinitely) stay in place with
`status: stale` and `archive_reason` set — only handoffs are archived
to a separate directory.

When one doc supersedes another, set `superseded_by: <slug>` on the
older doc and `supersedes: [<slug>, …]` on the newer. For partial
supersession, use `companion: [<slug>, …]` and explain the
relationship in `archive_reason`.

The HTML pages under `docs/prototype/*.html` are unaffected by this
convention and follow the separate rules above.
