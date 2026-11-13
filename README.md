# Language check

Enable this bundled plugin by adding `language_check` to the existing
`plugins.enabled` list in the active profile's config.yaml. Preserve other entries.
The checker defaults to enabled once the plugin is loaded.

```yaml
plugins:
  enabled: [language_check]
  entries:
    language_check:
      settings:
        enabled: true
        min_chars: 20
        inject_budget_percent: 5.0
        inject_max_chars: 0
        context_window_tokens: 128000
        judge_timeout_ms: 12000
        rewrite_timeout_s: 8
        log_path: ""
        judge_provider: zai
        judge_model: glm-5.3-flash
        rewrite_model: ""  # inherits judge_model
        disable_thinking: true
```

The judge receives only lines starting with a number and a period (`^\d+\.`)
from the discipline file, plus the `### 自动提炼` section of profile `SOUL.md`
when present (through the next heading of level 3 or above). Narrative and
unnumbered Bad Case examples are omitted. Rewriting still receives the full
discipline file. `judge_timeout_ms` sets the judge deadline in milliseconds;
`rewrite_timeout_s` sets the rewrite deadline in seconds.

Place the full constraints in `<hermes_home>/language-discipline.md`.
No core prompt or tool definitions are changed.

## Environment

Prefer `judge_provider` plus `judge_model` in the plugin settings above. Both
judge and rewrite use that provider; an empty `rewrite_model` inherits the judge
model. Bare model names are accepted; a matching `provider/` prefix is stripped.
The provider registry supplies its default URL and credential variable names.
The active profile's existing `providers:` entry overrides them, for example:

```yaml
providers:
  my-endpoint:
    base_url: https://llm.example.com/v1
    key_env: MY_LLM_API_KEY
plugins:
  enabled: [language_check]
  entries:
    language_check:
      settings:
        judge_provider: my-endpoint
        judge_model: my-model
```

`custom_providers:` list entries are also supported and take precedence over
`providers:` entries and builtin endpoints with the same name (case insensitive):

```yaml
custom_providers:
  - name: kimi-coding
    base_url: https://api.kimi.com/coding/
    api_key: ""
    api_key_env: KIMI_API_KEY
    api_mode: anthropic_messages
  - name: zai
    base_url: https://api.z.ai/api/coding/paas/v4
    api_key: ""
    api_mode: chat_completions
plugins:
  enabled: [language_check]
  entries:
    language_check:
      settings:
        judge_provider: kimi-coding
        judge_model: kimi-for-coding
```

`api` and `url` are also accepted endpoint keys. Credential resolution is inline
`api_key`, then process environment named by `api_key_env` (or `key_env`), then
provider registry environment variables, then Hermes `get_secret` for those same
names in that order. For example, Kimi uses `KIMI_API_KEY` / `KIMI_CODING_API_KEY`;
Z.AI uses `GLM_API_KEY` / `ZAI_API_KEY` / `Z_AI_API_KEY`.
An unmatched custom name falls back to the builtin registry.

`api_mode: chat_completions` sends `/chat/completions` with Bearer auth.
`api_mode: anthropic_messages` sends `/v1/messages` with `x-api-key` and
`anthropic-version: 2023-06-01`, a separate `system` field and user messages;
text response blocks are concatenated. A base already ending in `/v1` gets
`/messages` appended. When omitted, mode defaults to the registry mode or
`chat_completions` for an unknown custom name. Both modes use the same deadline
and one HTTP attempt. OAuth and Responses-only transports are not supported.
An invalid explicit provider fails open without sending credentials elsewhere.

Without `judge_provider`, the legacy environment path is preserved:
`ZAI_API_KEY` (process environment first, then `get_secret`) and
`LANGUAGE_CHECK_LLM_BASE_URL` (default `https://api.z.ai/api/coding/paas/v4`).
This path retains `provider/model` model syntax. Without credentials judging is
disabled, a warning is logged, and the original reply passes through.
`key_source` records `inline`, `environ:<variable>`, or `secret:<variable>` for
explicit providers; the legacy path records `environ`, `secret`, or `missing`.
`disable_thinking` defaults to true and adds `thinking: {type: disabled}` only
for `chat_completions` when the request model contains `glm` (case insensitive).

Settings are read through `ctx.get_config()` and the active configuration backend
on every reply. Change `plugins.entries.language_check.settings.enabled` to
`false` to bypass checking immediately without unloading the plugin. Initial
plugin activation requires normal plugin discovery (start a new process or use
the existing plugin reload operation). No reload is needed for the settings switch.

The hook holds streaming output until judging and optional rewriting finish.
It rewrites once and does not judge the rewrite. Missing/empty rules, invalid
judge JSON, provider errors and timeouts pass through the original response.
The framework's `plugins.hook_callback_timeout` (default 30 seconds) still applies;
setting it below the auxiliary call budgets can cause the framework to withhold
the reply. Each judge/rewrite has a plugin wall-clock deadline covering queueing and the single HTTP request. A context-copied worker runs the auxiliary call; the caller abandons its
result at the deadline and shuts down the executor with `wait=False`. The HTTP
request can finish in the background; Python cannot forcibly stop a running worker thread.

Every invocation appends a JSONL record, including skipped and failed checks.
The default is `<hermes_home>/logs/language-check.jsonl`. Records contain the
original reply and, when rewritten, the replacement, plus both stages' elapsed
milliseconds (including failed calls), verdict and error. `rules_chars` records
the character count sent to the judge, including on timeout; `error` retains
the exception type such as `TimeoutError`. Log failures only warn.


`wall_ms` in JSONL measures the hook through database persistence, including
configuration and rules loading (excluding the final JSONL append). PostgreSQL
stores a snapshot immediately before INSERT, including any connection/schema
cost but excluding its own INSERT/commit. `judge_ms` and
`rewrite_ms` measure their respective stages. `model` is the original reply model.

Optional PostgreSQL persistence reads `LANGUAGE_CHECK_PG_HOST`,
`LANGUAGE_CHECK_PG_PORT`, `LANGUAGE_CHECK_PG_DB`, `LANGUAGE_CHECK_PG_USER` and
`LANGUAGE_CHECK_PG_PASSWORD` from the profile environment. No credentials are
stored in settings. Install psycopg2 in the runtime environment if needed.
When HOST is absent only JSONL is used. Plugin registration creates the table and
indexes when configured; a lost connection is re-established on the next event.
The sink keeps at most one connection, closing it if profile credentials change.
Concurrent writes skip PostgreSQL when the connection is busy, keeping JSONL.

Each event uses one parameterized INSERT and commit; JSONL is written afterward
even when PostgreSQL fails.
Database failures only warn, close the connection and skip that event without
retrying; JSONL and the selected reply survive. Connection attempts have a 3s
connect timeout, statements/locks a 1s timeout, and TCP keepalives/user timeout
bound dead-connection detection where supported by libpq/the OS. Persistence is
synchronous: database connection failure can add up to the connection timeout to
reply latency, reflected in JSONL `wall_ms`. No connection pool or
background database worker is used.

Run these queries in `hermes_eval`:

```sql
-- Rates among evaluated replies (skip/errors with no verdict excluded).
SELECT count(*) AS evaluated,
       avg(violation::int) AS violation_rate,
       avg(rewritten::int) AS rewrite_rate
FROM language_check_events WHERE violation IS NOT NULL;

-- Transform and stage latency distributions, milliseconds.
SELECT percentile_cont(ARRAY[0.50, 0.95]) WITHIN GROUP (ORDER BY wall_ms) AS wall_p50_p95,
       percentile_cont(ARRAY[0.50, 0.95]) WITHIN GROUP (ORDER BY judge_ms) AS judge_p50_p95,
       percentile_cont(ARRAY[0.50, 0.95]) WITHIN GROUP (ORDER BY rewrite_ms)
         FILTER (WHERE violation) AS rewrite_p50_p95
FROM language_check_events WHERE skipped IS NULL;

SELECT session_id, count(*) AS events,
       count(*) FILTER (WHERE violation) AS violations,
       count(*) FILTER (WHERE rewritten) AS rewrites,
       count(*) FILTER (WHERE error IS NOT NULL) AS errors,
       avg(wall_ms) AS mean_wall_ms
FROM language_check_events GROUP BY session_id ORDER BY events DESC;
```

Dynamic rules are read from `<hermes_home>/language-rules-dynamic.md`, capped by the percentage budget (default 6400 characters) and optional `inject_max_chars` hard cap and cached by profile path and mtime. Missing or blank files add
no context. `pre_llm_call` returns a `context` directive every turn: the host appends
it to the current user message's API copy and carries it through that turn's
requests. This host hook runs once per turn, not before each provider attempt,
and does not accept replacement request messages. The host preserves injected
bytes in its `api_content` sidecar for stable historical replay; original string
message content is unchanged. No block directives are returned. Each invocation
logs `hook`, `ts`, `file_chars`, `truncated`, `window_tokens`, `budget_chars`, and `injected_chars` (file characters, excluding the prefix) to
the configured JSONL without PostgreSQL persistence. Failures add no context.

Run `scripts/language_rules_distill.py` explicitly to distill rules. The default
target is `<hermes_home>/language-rules-dynamic.md`; `DISTILL_DYNAMIC_PATH`
overrides it. Rules are deduplicated by name with at most 30 entries
(`--max-entries` defaults to 30 and may lower that cap). Provenance continues to
append to the discipline archive. SOUL.md writes require `--write-soul`;
`DISTILL_SOUL_PATH`, `DISTILL_ARCHIVE_PATH`, and `DISTILL_STATE_PATH` retain their
existing overrides. The JSON summary reports both `dynamic_bytes` and `soul_bytes`.

Injection keeps the newest contiguous complete lines within the character budget;
an oversized newest line is hard truncated. Warnings include dropped line counts
and repeat only after the file mtime/content changes. After committed compaction,
`post_compaction` warns and records `dynamic-rules-over-budget` when the file
exceeds the budget or contains at least 30 rule lines (`- `, excluding headings).
The observer payload has no summary text and cannot edit the compression summary.
Distill prints a WARNING before its JSON summary when a candidate is skipped
because the dynamic section is full; skipped reasons remain unchanged.

The injection budget is `context_window_tokens × inject_budget_percent / 100`,
using a conservative constant of 1.0 character per token. Defaults give 6400
characters (5% of 128000). The current host payload exposes neither the context
window nor an agent reference, so both hooks use the configurable
`context_window_tokens` fallback. Set it to the model window when known.
`inject_max_chars` optionally caps this budget; zero or omission means no hard cap.
Warnings report `window_tokens` and the effective `budget_chars` after the cap.
