# docs-tidy apply — warnings report
Generated: 2026-05-26

This report lists items that the apply step intentionally did **not** fix
automatically, per the migration plan's "flag, don't decide" guardrails.
Review and address each before closing out the refactor.

## Bare-filename refs to renamed files

When a file was renamed (e.g. `handoff-aloepri-attack-resistance.md` →
`2026-05-18-aloepri-attack-resistance.md`), references to the **old bare
filename** in backtick code spans were NOT rewritten automatically because
they are often used in contexts where the rename shouldn't propagate
(e.g. historical citations). Review each and decide:

- `docs/plans/path-1-gelo-gemma.md:938`
  - Has: `` `gemma4-architecture-roadmap.md` `` (in link text)
  - New filename: `2026-05-18-gemma4-architecture-support.md`
  - The link URL above was already rewritten; only the visible text is stale.

- `docs/handoffs/2026-05-19-alg2-qwen3-shape-analysis.md:324`
  - Has: `` `handoff-aloepri-quantisation-and-alg2-gaps.md` ``
  - New filename: `2026-05-21-aloepri-quantisation-and-alg2-gaps.md`

## Pre-existing broken absolute-path-as-relative links (not caused by this refactor)

These links were broken before the docs-tidy run — they use an
absolute-style path like `docs/plans/foo.md` as the URL inside a markdown
link, so they resolve incorrectly from any source location. The apply step
identified them as broken but did not rewrite them; investigate per file:

- `docs/handoffs/2026-05-21-aloepri-quantisation-and-alg2-gaps.md`
  - link → `docs/plans/path-2-aloepri-gemma.md`
  - link → `docs/handoffs/2026-05-21-path-2-aloepri-next-steps.md`
  - link → `docs/handoffs/2026-05-18-aloepri-attack-resistance.md` (×2)
- `docs/handoffs/2026-05-21-path-2-aloepri-next-steps.md`
  - link → `path-2-aloepri-gemma.md` (sibling-style, but file is in `../plans/`)

## Out-of-docs/ link (intentionally skipped)

- `docs/handoffs/2026-05-21-ima-transformer-paper-disparity.md`
  - link → `../../../.claude/projects/-home-timo-repos-private-rag/memory/feedback_no_cpu_for_gpu_workloads.md`
  - This is a reference to a Claude memory file outside the repo. Either drop the link or replace with a stable summary.

## Per-type required sections — coverage warnings

The skill specifies required section headers per doc type. The apply step
emits these as warnings rather than auto-inserting `<!-- TODO -->` stubs.
Spot-check the following classes; many docs use equivalent sections under
different names (e.g. an opening `**Status:** …` line in handoffs instead
of a `## Status` header):

- **handoffs (29 files)** required: `## Context`, `## Current state`,
  `## Known issues`, `## Next steps`, `## Owner`. Most existing handoffs
  use a mix of `**Status:**` line + thematic H2s rather than this template.
- **plans (14 files)** required: `## Objective`, `## Phases`/`## Milestones`,
  `## Open questions`, `## Status`.
- **prototype-notes (16 files)** required: `## Goal`, `## What we built`,
  `## Decisions made`, `## What we'd do differently`, `## Status`.
- **research (13 files)** generally use paper-summary structure
  (`## Citation`, `## Key contributions`, `## Relevance to project`,
  `## Open questions`) — coverage varies.

Recommendation: don't bulk-edit. When a doc is next touched substantively,
align section headers to the template.

## Cross-references inside HTML

`docs/prototype/*.html` were NOT scanned for cross-references to moved .md
files (HTML is out of the migration's automated cross-ref pass). The HTML
pages are the public artefact and per `CLAUDE.md` should minimise code/doc
references, so few or no broken refs are expected. Manual audit:

```
grep -rE 'docs/(prototype|research|plans)/[a-z0-9_-]+\.md' docs/prototype/*.html
```

## Type/folder mismatch decisions worth revisiting

A few files were placed in folders that don't strictly match their `type`.
Documented here so future drift is intentional:

- `docs/plans/private-inference-comparison-framework.md` is `type: reference`,
  kept in `plans/` (no dedicated `reference/` folder until critical mass).
- `docs/dev/prototype/private-graph-rag-variant-a.md` is `type: prototype-note`
  but reads partly as a plan. Co-located with related design docs in
  `dev/prototype/`.
- `docs/research/aloepri-keymat-variance.md` is `type: theory` (debugs an
  external paper's algorithm). Kept in `research/` rather than carving out
  `research/theory/`.

## Files left in `partial` status (need follow-through)

These files are flagged `partial` and have outstanding work:

- `docs/plans/q4-gpu-weights.md` — paused, awaits dGPU test.
- `docs/plans/m1-10-fused-permuted-attention.md` — scaffold landed;
  cached-generation wiring halted by causal-mask leak (see
  `m1-10-security-review.md`).
- `docs/dev/prototype/gelo-llm.md` — primitives landed; harness/decode
  KV cache deferred.
- `docs/handoffs/2026-05-18-aloepri-attack-resistance.md` — Phase 1
  landed; Phases 2–3 pending.
- `docs/handoffs/2026-05-19-m2-7-attack-findings.md` — partially
  superseded by the Option-C rerun + steps 0/1/2a docs (companion links
  set in frontmatter).
