# Handoff — strong-Π + chat_parser=epsilon workaround

**Date:** 2026-05-21
**Branch:** `path-2-aloepri-gemma` (uncommitted: obfuscator `--pi-include-specials`, path-2 client `_EPSILON_CHAT_PARSER`, capture-script payload field, fresh strong-Π 4B GGUF)
**Goal:** Close the ~293-pair specials/UNUSED structural leak by permuting all 151669 active tokens (strong-Π) and keep llama-server's `/completion` endpoint 100 % robust under the resulting multi-language gibberish output — **without patching llama.cpp**.

## What landed

1. **Obfuscator `--pi-include-specials` flag** (`python/path-2/obfuscate_qwen3_gguf.py`). When set, Π permutes all NORMAL/BYTE *and* CONTROL/USER_DEFINED/UNKNOWN/UNUSED tokens within the pi_active range, eliminating the identity-fixed corner.

2. **Strong-Π Qwen3-4B GGUF rebuilt** end-to-end: `untied-keymat-h128-pi-strong-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-bf16-native.gguf` + `.key.npz`. 9.7 GB bf16. Build took ~1 min.

3. **`chat_parser=epsilon` workaround** baked into:
   - `python/path-2/aloepri_client.py` — `_EPSILON_CHAT_PARSER` constant, sent on every `/completion` request body.
   - `evals/aloepri-attacks/m2_7/capture_hidden_states.py` — direct POST in the hidden-state capture path.
   - `evals/aloepri-attacks/m2_7/capture_token_streams.py` — direct POST in the token-stream capture path.

   No llama.cpp patch needed. Verified **65/65 corpus prompts pass** through both the raw HTTP path and through `AloePriClient.complete()`.

## Why this works (and the dead-ends ruled out)

Stock llama.cpp's `task_result_state::update_chat_msg` (`vendor/llama.cpp/tools/server/server-task.cpp:158-169`) runs `common_chat_parse` on every generated chunk for **all** completion endpoints including raw `/completion`. With strong-Π the cumulative de-tokenized text is multi-language gibberish, and on ~5–9 % of token sequences the default PEG grammar `content(rest()) + end()` at `common/chat.cpp:2494` throws "Failed to parse input at pos 0".

**Workarounds I tried and ruled out:**

| Approach | Result |
|---|---|
| `--no-jinja --reasoning off` server flags | doesn't bypass — `update_chat_msg` is unconditional |
| `--chat-template chatml` | doesn't bypass |
| `chat_format: 0/1/2` request field | doesn't bypass — just selects which PEG variant to build |
| `parse_tool_calls: False`, `reasoning_format: none` | doesn't bypass |
| Stream mode + bytes-decode iter_lines | helps marginally (91 → 95 % pass) but not 100 % |
| Older llama.cpp tag | rejected — would lose AloePri matrix-Γ kernel + tensor-dump patches |
| llama.cpp source patch (try/catch around `update_chat_msg`) | works (initially landed) but not needed; reverted |

**The actual fix:** `vendor/llama.cpp/tools/server/server-task.cpp:432` plumbs a `chat_parser` request field into `chat_parser_params.parser` via `arena.load(json)`. If the supplied arena is non-empty, `common_chat_peg_parse` uses ours instead of building the default `content(rest()) + end()`. We supply

```json
{"parsers":[{"type":"epsilon"}],"rules":{},"root":0}
```

The `epsilon` PEG primitive matches the empty prefix and never fails. `common_chat_parse` returns an empty `common_chat_msg`; `update_chat_msg` sees `new_msg.empty()` and falls through without throwing. The streamed `tokens` field is populated by an independent code path and is unaffected.

## Files modified (uncommitted)

- `python/path-2/obfuscate_qwen3_gguf.py` — `--pi-include-specials` flag + plumbing through `rewrite_gguf`
- `python/path-2/aloepri_client.py` — `_EPSILON_CHAT_PARSER` constant; request body now includes `chat_parser`
- `evals/aloepri-attacks/m2_7/capture_hidden_states.py` — same `chat_parser` field in the direct-POST body
- `evals/aloepri-attacks/m2_7/capture_token_streams.py` — same `chat_parser` field in the direct-POST body
- `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-strong-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-bf16-native.gguf` (+ `.key.npz`) — rebuilt artifact

vendor/llama.cpp submodule is **unchanged** — the try/catch patch I briefly applied is reverted (verified with `git diff --stat`).

## Next steps

1. **Re-measure 4B obfuscated cell of §08 with the strong-Π GGUF.** Spawn a new server pointed at the strong-Π artifact, re-run static + hidden + attn + token-stream captures. Expected: TFMA/SDA/NN/IA stay near plain floor; VMA might shift (more sources permuted = more sorted-quantile signal disruption); ISA/IMA-EmbedRow-ridge unchanged (ridge attack now retired per `aloepri-attacks.md`).
2. **8B strong-Π build + measurement.** Same flag, same flow.
3. **Then resume IMA-EmbedRow-transformer driver fix** (closed-form synthetic-ridge warm-start + public-corpus pipeline) — independent of this server-side work.

## Other dev tasks in flight

- **IMA-EmbedRow-transformer driver fix** — tracked separately; conceptual plan in `docs/research/aloepri-attacks.md`, optimizer diagnosis in `docs/handoffs/2026-05-20-ima-embedrow-transformer-investigation.md`.
