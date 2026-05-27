# docs-tidy migration plan
Generated: 2026-05-26
Revised: 2026-05-26 (post-grill)
Scope: `docs/` (74 .md files). Top-level `README.md`, `deploy/**/README.md`, and `evals/**/*.md` are code-adjacent and out of scope. `CLAUDE.md` is **in scope** for a small additive edit.

## Grilled decisions (10)

| # | Decision | Outcome |
|---|----------|---------|
| 1 | Folder layout | Split prototype: .md → `docs/dev/prototype/`, .html stays in `docs/prototype/` |
| 2 | Dev-log placement | Collect into `docs/dev/logs/` (not paired-with-plan) |
| 3 | Stale handling | Mark in-place via `status: stale`. **No** `docs/archive/` directory |
| 4 | Research/prototype boundary | `aloepri-qk-norm-matrix-gamma-threat-model.md` → `dev/prototype/`; `aloepri-keymat-variance.md` stays in `research/` |
| 5 | Handoff renames | Last-update-date prefix, drop `handoff-`/`-handoff` slug noise |
| 6 | Frontmatter scope | Rich: `type`, `status`, `created`, `updated`, `tags`, plus optional `supersedes`, `superseded_by`, `companion`, `archive_reason` |
| 7 | Section stubs | Warnings only — no TODO stubs auto-inserted |
| 8 | Cross-references | Mechanical sed for absolute paths + relative md links; bare filenames flagged for review |
| 9 | Supersession clusters | Leave separate, link via frontmatter; one body banner on the explicit-supersession stale doc |
| 10 | CLAUDE.md update | Add "Markdown docs (`docs/**/*.md`)" section to lock convention forward |

## Target folder layout

```
docs/
├── handoffs/         24 existing + 5 incoming = 29 dated .md files
├── plans/            ~10 active + 2 stale-in-place
├── research/         13 external-research notes + theory
├── prototype/        ONLY .html + _nav.js + css (public artefact)
└── dev/
    ├── prototype/    ~16 .md design notes
    └── logs/         2 dev-log files
```

No `docs/archive/`. No restructuring of `docs/handoffs/`, `docs/plans/`, or `docs/research/`. The `docs/prototype/` HTML and its references in `docs/index.html`, `docs/prototype/_nav.js`, and `CLAUDE.md` remain valid.

---

## Actions

### Create directories

- [ ] `mkdir -p docs/dev/prototype`
- [ ] `mkdir -p docs/dev/logs`

### Move + rename: plans → handoffs (3 files)

Use `git mv`; renames drop the `handoff-` prefix and add `YYYY-MM-DD-` (last-update date) prefix.

- [ ] `docs/plans/handoff-aloepri-attack-resistance.md` → `docs/handoffs/2026-05-18-aloepri-attack-resistance.md`
- [ ] `docs/plans/handoff-aloepri-quantisation-and-alg2-gaps.md` → `docs/handoffs/2026-05-21-aloepri-quantisation-and-alg2-gaps.md`
- [ ] `docs/plans/path-2-aloepri-next-steps.md` → `docs/handoffs/2026-05-21-path-2-aloepri-next-steps.md`

### Move + rename: prototype → handoffs (2 files)

- [ ] `docs/prototype/aloepri-gemma-handoff.md` → `docs/handoffs/2026-05-21-aloepri-gemma-deferred.md`
- [ ] `docs/prototype/gemma4-architecture-roadmap.md` → `docs/handoffs/2026-05-18-gemma4-architecture-support.md`

### Move: prototype .md → dev/prototype/ (12 files, no rename)

- [ ] `docs/prototype/aloepri-attack-harness.md` → `docs/dev/prototype/aloepri-attack-harness.md`
- [ ] `docs/prototype/aloepri-attack-harness-followups.md` → `docs/dev/prototype/aloepri-attack-harness-followups.md`
- [ ] `docs/prototype/caprise-two-party-kdf.md` → `docs/dev/prototype/caprise-two-party-kdf.md`
- [ ] `docs/prototype/dp-forward.md` → `docs/dev/prototype/dp-forward.md`
- [ ] `docs/prototype/future-rnd.md` → `docs/dev/prototype/future-rnd.md`
- [ ] `docs/prototype/gelo-complexity-analysis.md` → `docs/dev/prototype/gelo-complexity-analysis.md`
- [ ] `docs/prototype/gelo-llm.md` → `docs/dev/prototype/gelo-llm.md`
- [ ] `docs/prototype/gelo.md` → `docs/dev/prototype/gelo.md`
- [ ] `docs/prototype/inference-optimization.md` → `docs/dev/prototype/inference-optimization.md`
- [ ] `docs/prototype/private-graph-rag-variant-a.md` → `docs/dev/prototype/private-graph-rag-variant-a.md`
- [ ] `docs/prototype/remote-rag.md` → `docs/dev/prototype/remote-rag.md`
- [ ] `docs/prototype/reranking.md` → `docs/dev/prototype/reranking.md`

### Move: research → dev/prototype/ (4 files, no rename)

These are own-system design / threat-model docs misfiled under research.

- [ ] `docs/research/hd3-non-pow2-fix.md` → `docs/dev/prototype/hd3-non-pow2-fix.md`
- [ ] `docs/research/private-graph-rag-design.md` → `docs/dev/prototype/private-graph-rag-design.md`
- [ ] `docs/research/private-rag-system-design.md` → `docs/dev/prototype/private-rag-system-design.md`
- [ ] `docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md` → `docs/dev/prototype/aloepri-qk-norm-matrix-gamma-threat-model.md`

### Move: dev-logs → dev/logs/ (2 files, no rename)

- [ ] `docs/plans/path-2-status.md` → `docs/dev/logs/path-2-status.md`
- [ ] `docs/prototype/aloepri-attack-harness-findings.md` → `docs/dev/logs/aloepri-attack-harness-findings.md`

### Mark stale in-place (3 files, no move)

Frontmatter only; files stay where they are.

- [ ] `docs/plans/m1-10-phase4-findings.md` — `status: stale`, `archive_reason: "Phase 2 deprecated post-F1+ design decision"`
- [ ] `docs/plans/m1-12-permuted-attention-batched-decode.md` — `status: stale`, `archive_reason: "R1.4 Phase A aborted; failed gate by 16× on iGPU. Bucket 2 deferred indefinitely."`
- [ ] `docs/handoffs/2026-05-19-aloepri-attack-surface-followups.md` — `status: stale`, `superseded_by: 2026-05-20-aloepri-attacks-status-and-paired-data-defences`, `archive_reason: "threads 1-3 superseded; threads 4+ retained for context"`
  - Plus body banner at top: a markdown blockquote pointing readers to the superseding doc.

### Add frontmatter to all .md files

Every `.md` file under `docs/` (post-moves) gets a YAML frontmatter block. Schema:

```yaml
---
type: <handoff | plan | prototype-note | research | theory | dev-log | reference>
status: <current | partial | stale>
created: YYYY-MM-DD              # from `git log --diff-filter=A --format=%ci -- <file> | tail -1`
updated: YYYY-MM-DD              # from `git log -1 --format=%ci -- <file>`
tags: [<auto-populated where obvious>]
# Optional fields, present only when applicable:
# supersedes: <slug>
# superseded_by: <slug>
# companion: [<slug>, ...]
# archive_reason: "..."
---
```

Status assignments derived from the audit:

| Status | Count | Files |
|--------|------:|-------|
| `current` | ~65 | Most handoffs, active plans, all research, all dev/prototype |
| `partial` | ~5 | `q4-gpu-weights.md` (paused, awaits dGPU); `m1-10-fused-permuted-attention.md` (cached-gen wiring halted by causal-mask leak); `gelo-llm.md` (primitives landed, harness deferred); `handoff-aloepri-attack-resistance.md` (Phase 1 landed, 2-3 pending); `2026-05-19-m2-7-attack-findings.md` (partially superseded by Addendum) |
| `stale` | 3 | The 3 in-place marked above |

Tag auto-population uses the obvious topic identifiers found in filename/title: `aloepri`, `gelo`, `caprise`, `m1.10`, `m1.11`, `m1.12`, `m2.7`, `path-1`, `path-2`, `qwen3`, `gemma`, `q4`, `bf16`, `hd3`, `attention`, `attack`, `reranking`, `pir`, `graph-rag`. Frontmatter `tags: []` left empty if no obvious match.

Supersession/companion cross-links (frontmatter fields):

| Doc | Field | Value |
|-----|-------|-------|
| `2026-05-19-aloepri-attack-surface-followups.md` | `superseded_by` | `2026-05-20-aloepri-attacks-status-and-paired-data-defences` |
| `2026-05-20-aloepri-attacks-status-and-paired-data-defences.md` | `supersedes` | `[2026-05-19-aloepri-attack-surface-followups]` |
| `2026-05-19-m2-7-attack-findings.md` | `companion` | `[2026-05-19-option-c-m2-7-rerun-findings, 2026-05-19-option-c-steps-0-1-2a-findings]` |
| `2026-05-19-option-c-m2-7-rerun-findings.md` | `companion` | `[2026-05-19-m2-7-attack-findings, 2026-05-19-option-c-steps-0-1-2a-findings]` |
| `2026-05-19-option-c-steps-0-1-2a-findings.md` | `companion` | `[2026-05-19-m2-7-attack-findings, 2026-05-19-option-c-m2-7-rerun-findings]` |
| `2026-05-20-ima-embedrow-transformer-investigation.md` | `companion` | `[2026-05-21-ima-transformer-paper-disparity]` |
| `2026-05-21-ima-transformer-paper-disparity.md` | `companion` | `[2026-05-20-ima-embedrow-transformer-investigation]` |
| `2026-05-21-attn-offload-spike.md` | `companion` | `[2026-05-21-gelo-perf-shield-attn-batched]` |
| `2026-05-21-gelo-perf-shield-attn-batched.md` | `companion` | `[2026-05-21-attn-offload-spike]` |
| `path-2-aloepri-gemma.md` | `companion` | `[path-2-status]` (running log, now in dev/logs/) |

### Missing-section reporting (no auto-insert)

After frontmatter is added, emit a per-file warning list of missing per-type required section headers (per the skill's section spec). No `<!-- TODO -->` stubs inserted into files. The warning report is written to `docs-tidy-warnings.md` next to the plan file. The user can review and decide per-file whether to add sections.

### Cross-reference updates

Mechanical sed pass for the moves above. Two patterns updated:

1. **Absolute repo paths in backtick code spans**:
   - `docs/research/hd3-non-pow2-fix.md` → `docs/dev/prototype/hd3-non-pow2-fix.md`
   - `docs/research/private-graph-rag-design.md` → `docs/dev/prototype/private-graph-rag-design.md`
   - `docs/research/private-rag-system-design.md` → `docs/dev/prototype/private-rag-system-design.md`
   - `docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md` → `docs/dev/prototype/aloepri-qk-norm-matrix-gamma-threat-model.md`
   - `docs/plans/handoff-aloepri-attack-resistance.md` → `docs/handoffs/2026-05-18-aloepri-attack-resistance.md`
   - `docs/plans/handoff-aloepri-quantisation-and-alg2-gaps.md` → `docs/handoffs/2026-05-21-aloepri-quantisation-and-alg2-gaps.md`
   - `docs/plans/path-2-aloepri-next-steps.md` → `docs/handoffs/2026-05-21-path-2-aloepri-next-steps.md`
   - `docs/prototype/aloepri-gemma-handoff.md` → `docs/handoffs/2026-05-21-aloepri-gemma-deferred.md`
   - `docs/prototype/gemma4-architecture-roadmap.md` → `docs/handoffs/2026-05-18-gemma4-architecture-support.md`
   - `docs/plans/path-2-status.md` → `docs/dev/logs/path-2-status.md`
   - `docs/prototype/aloepri-attack-harness-findings.md` → `docs/dev/logs/aloepri-attack-harness-findings.md`
   - Plus all 12 `docs/prototype/*.md` → `docs/dev/prototype/*.md`.

2. **Relative markdown links** (e.g. `[name](../prototype/gemma4-architecture-roadmap.md)`):
   - For files moving across folder depth, update the `../` prefix as required.
   - Confirmed instances live in `docs/plans/path-1-gelo-gemma.md` (links to `../prototype/gemma4-architecture-roadmap.md`).

3. **Bare-filename backtick refs** (e.g. `` `path-2-status.md` `` without folder prefix):
   - **Not** rewritten automatically. The renames in this plan keep filenames stable in 13 of 19 cases (only the 5 handoff renames change the filename), so bare-name refs are mostly unaffected. The 5 renamed files (handoff-aloepri-attack-resistance.md, etc.) need a separate pass — emit them in a `docs-tidy-warnings.md` review list.

Apply will print:
```
[CROSS-REF] Updated N absolute-path references.
[CROSS-REF] Updated M relative markdown-link references.
[REVIEW] K bare-filename references found referring to renamed files
         (see docs-tidy-warnings.md).
```

### CLAUDE.md update

Append a new section to `CLAUDE.md` (sibling to the existing "HTML docs" section). Content:

```markdown
## Markdown docs (`docs/**/*.md`)

All markdown docs require YAML frontmatter:

​```yaml
---
type: <handoff|plan|prototype-note|research|theory|dev-log|reference>
status: <current|partial|stale>
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: []
# Optional: superseded_by, supersedes, companion, archive_reason
---
​```

Folder mapping (by `type`):
- `handoff`        → `docs/handoffs/YYYY-MM-DD-<slug>.md` (filename date = last update)
- `plan`           → `docs/plans/`
- `prototype-note` → `docs/dev/prototype/`
- `research`       → `docs/research/`
- `theory`         → `docs/research/`
- `dev-log`        → `docs/dev/logs/`
- `reference`      → `docs/plans/` (no dedicated folder until critical mass)

When a doc becomes stale (superseded, aborted, paused indefinitely), set
`status: stale` and add `archive_reason`. Do **not** move the file to an
archive directory — keep it in place so historical chains stay co-located
with the work that superseded them.

When one doc supersedes another, set `superseded_by` on the older doc and
`supersedes: [...]` on the newer. For partial supersession, use
`companion: [...]` instead and explain the relationship in `archive_reason`.

The HTML pages under `docs/prototype/*.html` are unaffected by this
convention and follow the separate rules above.
```

(The CLAUDE.md update itself is a single Edit.)

### What is intentionally NOT done

- No `docs/archive/` directory created. Stale plans live in `docs/plans/` with `status: stale`.
- No content merging of the 4 supersession clusters. Cross-links carried in frontmatter only.
- No section-header stubs inserted. Missing sections reported, not patched.
- No moves of `bench-results/` (repo root) into `docs/dev/bench/`. Out of scope; revisit separately.
- No changes to `docs/prototype/*.html`, `docs/prototype/_nav.js`, `docs/index.html`. They remain valid.
- No moves of `evals/**/*.md` (code-adjacent, out of scope).
- No new `docs/README.md` index. Add if/when the corpus needs it.

## Summary counts (if applied)

| Action | Count |
|--------|------:|
| New directories | 2 (`docs/dev/prototype/`, `docs/dev/logs/`) |
| Files moved (no rename) | 18 (12 prototype/.md + 4 research/→dev/prototype/ + 2 dev-log) |
| Files renamed + moved | 5 (the handoffs incoming) |
| Files marked stale in-place | 3 |
| Frontmatter blocks added | 74 |
| Cross-reference updates (estimated) | ~30 absolute-path + ~4 relative-link |
| Bare-name refs flagged for review | ~6 |
| CLAUDE.md edits | 1 (append section) |
| Files deleted | 0 |

**Total files touched: ~78** (74 frontmatter + ~4 receiving cross-ref-only edits). Above the 20-file confirmation threshold — `apply` MUST prompt before executing.

## Pre-apply checklist

- [ ] Confirm `git status` is clean on the working branch.
- [ ] Confirm we're on a dedicated branch (the worktree is on `docs/refactor`, OK).
- [ ] Stash or commit any pending work before apply.
- [ ] Apply proceeds in this order: (1) mkdir new dirs; (2) `git mv` files (no rename batch); (3) `git mv` renames; (4) frontmatter pass; (5) in-place stale marking; (6) cross-ref sed pass; (7) CLAUDE.md edit; (8) write warnings report.
- [ ] After apply: `git status`, review diff, `cargo doc --no-deps` (if applicable) to verify no broken rustdoc intralinks, then commit.
