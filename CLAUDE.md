# Repository conventions for Claude

This file captures project-wide rules that apply regardless of which
file or crate is being edited. Loaded into context automatically.

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
- **Formatting:** run `npx prettier --write <file>` after every edit
  (per memory `feedback_prettier_after_html_edits.md`). HTML edits
  that skip prettier produce hard-to-review noise in subsequent
  diffs.

When in doubt: would this paragraph make sense to a reader who has
never seen the source tree? If it depends on knowing the crate
layout or having `cargo doc` open, it belongs in code/rustdoc, not
the HTML.
