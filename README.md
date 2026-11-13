# language_check

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that enforces **language discipline** on every reply — with a self-improving loop that distills new rules from violation data and injects them into every session dynamically.

## How it works

Three hooks, one closed loop:

```
            ┌──────────────────────────────────────────────┐
            │                                              │
  pre_llm_call        transform_llm_output      distill cron (daily)
  inject dynamic  →   judge reply (flash  │  ──→  aggregate violations in PG
  rules into every     LLM, ~2s), rewrite   │     distill high-confidence rules
  request tail         violations, fail-open │     into language-rules-dynamic.md
            └──────── post_compaction: warn when rules exceed budget ──┘
```

1. **Dynamic rule injection** (`pre_llm_call`): appends your rule file to the *tail* of each request (user-message suffix), so rules reach **running sessions immediately** without touching the frozen system prompt or breaking KV prefix cache. Budget = 5% of context window by default (`inject_budget_percent`), whole-line truncation, newest rules win.
2. **Reply gate** (`transform_llm_output`): every final reply is judged by a fast model (default `glm-5.3-flash`, thinking disabled) against your rules; violations are rewritten in-place (one retry), everything logged to JSONL + PostgreSQL. Fails open — a broken checker never blocks a reply.
3. **Over-budget alerts** (`post_compaction` + gateway log, deduped by file version): warns when the rule file exceeds the injection budget or the entry cap, including right after context compression.

`language_rules_distill.py` (standalone script, run from cron): aggregates violation events from PostgreSQL, and when the same rule trips ≥3 times distills a one-line rule — `禁X："real counter-example"→改：fix` — into the dynamic rule file. Capped, idempotent, `--dry-run` supported.

## Install

```bash
hermes plugin install   # from this repo / catalog
```

## Configuration (plugin settings)

| key | default | meaning |
|---|---|---|
| `enabled` | `true` | instant kill-switch, no restart |
| `min_chars` | `20` | replies shorter than this skip the judge |
| `judge_timeout_ms` | `12000` | hard budget; timeout = fail open |
| `inject_budget_percent` | `5.0` | % of context window allowed for injected rules |
| `inject_max_chars` | `0` | optional hard cap in chars (0 = none) |
| `context_window_tokens` | `128000` | fallback when the hook payload lacks the window |

## Environment

| var | meaning |
|---|---|
| `ZAI_API_KEY` | judge/rewrite model key (any OpenAI-compatible endpoint via `LANGUAGE_CHECK_LLM_BASE_URL`) |
| `LANGUAGE_CHECK_PG_HOST/PORT/DB/USER/PASSWORD` | PostgreSQL eval store (optional; JSONL fallback always written) |
| `LANGUAGE_RULES_FILE` | dynamic rule file path (default `<hermes_home>/language-rules-dynamic.md`) |

## Data

Events land in `language_check_events` (violation, violated_rules, judge analysis, severity, original/rewritten text, latency, diagnostics) — query examples in the plugin README.

## License

MIT
