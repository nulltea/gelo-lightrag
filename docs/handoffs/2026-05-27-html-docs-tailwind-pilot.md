---
type: handoff
status: current
created: 2026-05-27
updated: 2026-05-27
tags: [docs, prototype, tailwind]
---

# Handoff — 2026-05-27 — HTML docs migration to Tailwind

A pilot Tailwind-CDN port of `docs/prototype/gelo-llm.html` landed
this session (commit `b6ac6d2`). It lives next to the original at
`docs/prototype/gelo-llm.tailwind-pilot.html` and is the reference
template for porting the remaining seven prototype docs. The user has
not yet decided whether to swap files in place; the pilot is for
side-by-side evaluation.

## What lives where

| Doc | Path | Bytes (gelo-llm) |
|---|---|---|
| Original | `docs/prototype/gelo-llm.html` | 222 KB · uses `css/site.css` + `_nav.js` |
| Pilot | `docs/prototype/gelo-llm.tailwind-pilot.html` | ~270 KB · self-contained Tailwind CDN |
| Site styles (legacy) | `docs/prototype/css/site.css` | 1 190 lines · still used by the other 7 docs |
| Shared nav (legacy) | `docs/prototype/_nav.js` | the cross-page topnav script |

## Pilot design choices — these are the decisions a future agent must
not relitigate without checking with the user first

1. **Tailwind via CDN only**, no `site.css`, no `_nav.js`. The user
   explicitly chose "Full Tailwind-CDN rewrite" over "borrow only
   structural patterns" and over a hybrid. Trust-zone color tokens
   (`--tee`, `--gpu`, `--client`, `--ink`, `--paper`, etc.) are
   re-declared inline in a scoped `<style>` block so SVG diagrams
   continue to render unchanged.
2. **System UI font stack** (Tailwind default `font-sans`).
   The user **rejected** loading JetBrains Mono / Fraunces from
   Google Fonts when I tried it (see exchange around "Why did you
   changed the font, rollback"). Future ports must not reintroduce
   webfonts.
3. **Body weight 450 / strong 650** via inline `<style>` rule on
   `main p, main li, main dd`. Bumped from default 400 because system
   UI fonts render thin at body sizes; weights are calibrated to land
   below medium (500). Don't push higher.
4. **Font sizes nudged +0.1rem** for `text-base` / `text-sm` /
   `text-xs` scoped to `main`. Same inline `<style>` rule.
5. **Page width `max-w-[1240px]`** matching the original
   `.sheet { max-width: 1240px }`. Don't shrink to Tailwind's
   `max-w-5xl` (1024px) — the architecture-review template I used as a
   starting point ran at 5xl but it's too narrow for these dense
   docs.
6. **Standardised component-chip palette** (§05 in the pilot, lines
   ~1100–1130). Reuse these exact classes in every other doc that
   uses category chips:

   | Category | Tailwind | Use for |
   |---|---|---|
   | Model | `bg-violet-100 border-violet-300 text-violet-900` | model identity / architecture variant |
   | Kernel | `bg-teal-100 border-teal-300 text-teal-900` | compute primitive (matmul, norm, attn) |
   | State | `bg-slate-200 border-slate-400 text-slate-800` | mutable runtime state (caches, buffers) |
   | Algorithm | `bg-amber-100 border-amber-300 text-amber-900` | orchestration / control flow / sampling |
   | Protocol | `bg-rose-100 border-rose-300 text-rose-900` | trust-boundary primitives (mask / shield / verify) |
   | Scheme | `bg-blue-100 border-blue-300 text-blue-900` | cryptographic / encoding scheme (AEAD, RATLS frame) |

7. **Gemma 4 dropped** from the gelo-llm pilot per the user's "Drop
   all gemma related stuff" instruction. The original doc still has
   the Gemma 4 collapsible block + reference diagrams + glossary
   rows; the user wants those gone in the Tailwind version. Future
   ports of *other* docs may keep their Gemma content unless the user
   says otherwise — this was a per-doc instruction.
8. **SVG diagrams preserved verbatim**, not reimplemented in Tailwind
   utility classes. The four hand-tuned diagrams in `gelo-llm.html`
   are ~500 lines of pixel-precise engineering art each;
   reimplementation in utility classes was attempted-and-rejected as
   "preserve them and inline the small CSS-variable vocabulary they
   reference." The scoped `<style>` block defines `.flow .zone-band-tee`,
   `.box-tee`, `.arrow-masked`, etc. Apply the same approach to other
   doc SVGs.
9. **Inline sticky nav** mirroring `_nav.js` items (Storage,
   Embedding, Reranking, Generation dropdown {GELO LLM, AloePri LLM},
   GraphRAG). No JS dep; uses Tailwind `group-hover` /
   `group-focus-within` for the dropdown. Current-page chip in
   emerald. Lives right under `<body>`, before `<main>`.

## What's left to port

Seven prototype docs are still on the legacy `site.css` + `_nav.js`
stack:

| Doc | Path | LOC (HTML) | Notes |
|---|---|---|---|
| Landing | `docs/prototype/index.html` | 50 KB | Has the headline stat strip + section grid pattern |
| Embedding | `docs/prototype/embedding.html` | 60 KB | Upstream protocol — sibling to gelo-llm |
| Reranking | `docs/prototype/reranking.html` | 94 KB | Sibling protocol; longest after the LLM pages |
| GraphRAG | `docs/prototype/graphrag.html` | 81 KB | Compass-side |
| AloePri LLM | `docs/prototype/aloepri-llm.html` | 205 KB | The largest doc; sibling of gelo-llm in the Generation dropdown |
| Storage | `docs/prototype/storage.html` | 34 KB | CAPRISE storage layer |
| Storage RemoteRAG | `docs/prototype/storage-remoterag.html` | 30 KB | Variant |

Order suggestion: **embedding → reranking** first (they share idioms
with gelo-llm and the user can compare three Generation/Protocol
sibling docs side by side), then **index** (the landing page sets the
nav-target for all routes), then **storage** + **storage-remoterag**
(smaller, easier wins), then **graphrag** + **aloepri-llm** (largest,
do last when the template is stable).

## Caveats / open decisions

- **No swap-in yet.** The pilot file ends in `.tailwind-pilot.html`.
  The user has not authorised renaming the file over the original.
  Do not move/delete `gelo-llm.html` without explicit go-ahead.
- **Nav linking.** The nav in the pilot links to `gelo-llm.html`,
  `embedding.html`, etc. — i.e. the *legacy* file names. If the
  user decides to swap each doc as it's ported, the nav will need
  updating per-page or via a shared template.
- **`_nav.js` lifecycle.** Once the last legacy doc is ported, the
  shared topnav script becomes dead. Until then it stays.
- **`css/site.css` lifecycle.** Same story — load-bearing for every
  unported doc.
- **Architecture-review prototype** (`docs/prototype/architecture-review-20260527-073711.html`,
  added in commit `5733e9c`) was the original Tailwind-CDN template
  the pilot derived from. Don't unify with it — the architecture
  review is a one-off report; the pilot is a doc template.

## Suggested next steps (next agent)

1. **Sanity check the pilot in a browser.** Open both
   `docs/prototype/gelo-llm.html` and `docs/prototype/gelo-llm.tailwind-pilot.html`
   side-by-side and confirm the visual feel matches what the user
   asked for. Bias to fix small things in the pilot before
   replicating its patterns onto the other docs.
2. **Pick the next doc** with the user — order suggestion above.
3. **Apply the same translation pattern** as gelo-llm: hero +
   stat strip + ToC + sectioned content + standardised component
   chips + sticky nav. Reuse the inline `<style>` block from the
   pilot verbatim where it covers SVG tokens, weight bumps, and size
   nudges.

## Suggested skills for the next session

- `frontend-design:frontend-design` — for visual judgement calls on
  the remaining ports.
- `docs-tidy` — once two or three docs are ported, run a quick audit
  to make sure cross-references between docs (`href="..."`) still
  resolve.
- No need for `code-review` / `verify` — these are static HTML
  changes with no test surface.

## Related artefacts

- Pilot file: `docs/prototype/gelo-llm.tailwind-pilot.html`
- Original: `docs/prototype/gelo-llm.html`
- Architecture-review template (origin of the visual idiom):
  `docs/prototype/architecture-review-20260527-073711.html`
- Pilot commit: `b6ac6d2 docs(prototype): add gelo-llm Tailwind-CDN style pilot`
- Same-session refactor commit (unrelated but landed first):
  `5733e9c refactor: deepen reranker trait, KG store, BERT provisioning; rename RayonCpuEngine`
