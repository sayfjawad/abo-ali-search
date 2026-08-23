# ADR: 001 - LLM Backend: GPT-OSS-120B on c4130-4xv100 instead of local Qwen3-8B

* **Status:** Accepted
* **Date:** 2026-08-23
* **Author:** Sayf Jawad (with Claude)

## 1. Context and Problem Statement

`/api/ask` auto-discovers a `scrib-r-backend-llama-1` Docker container on
`localhost` (`llm_base_url()` in `app.py`) and defaults `LLM_MODEL_ID` to
`qwen3-8b`. This app runs on `c4130-4xv100`, which also runs scrib-r's own
production stack — so it was, by construction, always talking to whichever
model that node's `llama` compose service happened to be serving: originally
`Qwen2.5-14B-Instruct`, a model the operator judged "niet slim genoeg voor
serieus werk" for scrib-r's own use, and never specifically validated for
this app's Arabic RAG use case either.

Separately, `c4130-4xv100` was migrated the same day (see scrib-r's own
ADR-009) to run GPT-OSS-120B — a materially stronger model — split across 3
of its 4 GPUs, replacing Qwen2.5-14B for scrib-r's own reasoning-worker and
`gemeente-search` (see that project's ADR-013/ADR-014).

## 2. Decision

Point this app at the same GPT-OSS-120B endpoint (`LLM_MODEL_ID=gpt-oss-120b`,
still via the existing auto-discover mechanism — no `LLM_BASE_URL` override
needed, both this app and the `llama` container live on the same machine).
Two code changes were required, not just a config/model swap, because the
existing code carried Qwen3-specific assumptions:

1. **Removed the `/no_think` suffix appended to every user message.** This is
   a Qwen3 inline chat-template convention with no effect on GPT-OSS's
   harmony reasoning format — it was silent dead weight even before this
   change (this app's actual serving model was already Qwen2.5-14B, not
   Qwen3, so it was arguably never doing anything).
2. **Added `reasoning_effort: "low"` to the request body**, harmony's actual
   mechanism for controlling how much a GPT-OSS model thinks before
   answering — confirmed live during the c4130 migration
   (`content` and `reasoning_content` come back as separate fields;
   `reasoning_effort: "low"` keeps the response short without starving
   `content`).
3. **Guarded against empty `content`.** Reasoning models can spend their
   entire token budget on `reasoning_content` and return `content: ""` —
   previously this app would silently show a blank answer. Now: if `content`
   is empty and `reasoning_content` is present, it's surfaced as
   `error: "empty_answer_reasoning_only"` instead.

## 3. Consequences

### 3.1. Pros
* Materially stronger model for a task (Arabic RAG over religious/political
  content, citation-grounded) that benefits from it — validated live post-
  switch with a real question, correct citations, coherent Arabic answer.
* The three code fixes above are model-family-correct now, not carried over
  unexamined from whatever the previous default happened to be.
* No new infrastructure dependency introduced — this app already ran on the
  same physical machine as the `llama` container; only the model file being
  served there changed.

### 3.2. Cons
* Still depends on scrib-r's `llama` compose service being up and serving
  the expected model — this app has no independent LLM of its own, and
  `llm_base_url()`'s auto-discovery silently returns `None` (not an error)
  if the container can't be found, degrading `/api/ask` to
  `{"answer": None, "error": "no_llm"}` without further diagnosis.
* `LLM_MODEL_ID` default in code (`gpt-oss-120b`) and the actually-served
  model are two independently-changeable things with no runtime check that
  they match — a future model swap on c4130 without a matching code/env
  update here would silently mislabel every response.

## 4. Alternatives Considered

1. **Keep the auto-discovered model as-is, whatever scrib-r happens to be
   running.** This was the implicit status quo. Rejected as a *decision* —
   accidentally inheriting scrib-r's model choice, with Qwen3-specific code
   quirks nobody had re-validated against whatever model was actually
   running, is exactly the failure mode this ADR fixes.
2. **Point at an external API (e.g. Claude, DeepSeek) instead of the local
   node**, matching the pattern `gemeente-search` used temporarily (see its
   ADR-013). Rejected: no cost or dependency benefit once a strong local
   model was available on the same physical machine.

## 5. References
* [`app.py`](../../app.py) — `llm_base_url()`, `call_llm` request body in
  `api_ask()`
* scrib-r ADR-009 (`architecture/07-decision-records/ADR-009-c4130-gpt-oss-120b.md`
  in the `scrib-r` repo) — the underlying c4130/GPT-OSS-120B migration
* `gemeente-search` ADR-013/ADR-014 — the same-day, same-model migration for
  a sibling project, including the `enable_thinking` vs. `reasoning_effort`
  correction referenced above
