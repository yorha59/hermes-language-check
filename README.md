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

## Jev shadow comparison

Optional settings under `plugins.entries.language_check.settings`:

```yaml
jev_shadow_enabled: false
jev_model: jev-1.14
jev_timeout_ms: 3000
jev_confidence_threshold: 0.75
```

Only `JEV_API_KEY` is read (environment, then profile `get_secret`). Shadow is
opt-in. After the flash path and its event persistence complete, a context-copied
daemon worker sends one request to
`https://mediocre-new-api.midway.run/typesafe/v1/systemone`, with no retries.
The state is the original reply body; numbered lines from the exact slimmed
flash rules become `r1`, `r2`, … Choice questions with `violation`/`ok` criteria.
Only violation choices meeting the confidence threshold count as high confidence.
The result never selects a reply, triggers rewriting, or changes fail-open.

The worker enforces a wall deadline as well as the HTTP timeout. A running HTTP
thread cannot be forcibly killed and may finish later; late results are ignored.
JSONL appends a separate `hook: jev_shadow` event with the original `turn_id`,
`session_id`, and `ts`, nullable `jev_*` fields, and successful per-rule
`jev_results` (choice mapped to a boolean, confidence, probabilities).
Disabled/skipped transforms keep these fields null. A failed flash judge has
no boolean verdict, so disagreement remains null. Errors record only their type.
`jev_wall_ms` excludes shadow persistence. No usage/cost estimate is inferred.

PG migrates with `ADD COLUMN IF NOT EXISTS`; the worker uses its own connection
and updates the latest row matching **turn_id and the original ts**, ordered by
ts then id. This avoids overwriting a different session that reused a turn id.
It updates only shadow fields and never inserts a duplicate transform event.
If the original PG insert was skipped/failed, no row is synthesized; JSONL still
retains the shadow result. Process exit can drop daemon work.

```sql
SELECT ts, session_id, turn_id, violation AS flash_violation,
       jev_high_confidence_rules, jev_wall_ms
FROM language_check_events
WHERE jev_status = 'shadow_ok' AND jev_disagrees_with_flash IS TRUE
ORDER BY ts DESC;
```

## Jev → GLM training and daily calibration

`jev_training_enabled` defaults to `false`. When true, each enabled final-reply
check starts a Jev task before GLM Flash and runs both concurrently, including
replies below `min_chars`. Flash judge/rewrite remains the sole user-visible
decision source. The reply never waits for Jev; its model latency critical path
is Flash judging and optional rewriting (plus existing event persistence).
Training takes precedence over `jev_shadow_enabled`, so no second shadow request
is sent. With training off, existing opt-in shadow behavior is unchanged.

The initial Flash event has `jev_training_status=pending`. Only after its PG
INSERT/commit attempt and JSONL append finish does a background collector persist
Jev results: it updates the same PG row identified by turn_id and original ts,
and appends a `hook: jev_training` JSONL event with the same identity. A Jev result
that finishes first still cannot update before the initial INSERT. Failed/skipped
inserts do not synthesize rows. Process exit can drop background collection.

Training records `jev_results` (per-rule choice, violation, confidence and both
probabilities), `jev_conclusion`, `jev_overall_confidence`,
`jev_overall_probability`, and `jev_training_status`. Final statuses are `trained`,
`unconfigured`, `timeout`, or `error`; none changes Flash or rewriting. Invalid
results are errors. No key or future is recorded. PostgreSQL adds nullable columns
idempotently; shadow updates also persist their per-rule results.

Aggregation never uses `jev_confidence_threshold`: any violation choice means
`jev_conclusion=true`, taking the maximum confidence and maximum violation
probability among violation choices. When every choice is ok, conclusion is false,
taking the minimum confidence and minimum ok probability across all rules.

Run the calibration manually (this does not install a cron job):

```sh
./venv/bin/python scripts/jev_calibrate.py --dry-run
./venv/bin/python scripts/jev_calibrate.py --output /path/to/jev-calibration.json
```

The default output is `<hermes_home>/jev-calibration.json`. The script only reads
PG, in a read-only transaction, using the existing `LANGUAGE_CHECK_PG_*`
credential resolver. It selects only timestamps, event IDs, labels and scores;
no reply text is read or printed. An old database without the migration reports
`pg_unavailable`; the calibrator never migrates it. `--dry-run` writes nothing.
Normal runs atomically replace the report and its adjacent `.state.json` sidecar.
Stdout starts with a delivery-friendly summary followed by the complete JSON.

Only trained events with non-null Flash truth, Jev conclusion and valid scores
are eligible. Confidence and probability are scanned independently at thresholds
0.00 through 0.99. Each threshold reports coverage, TP/FP/TN/FN, agreement, both
predicted-class precisions and their two-sided 95% Wilson lower bounds.
`--target-precision` defaults to 0.98; `--min-predicted-class-samples` defaults to
50 **in each predicted class**. The lowest threshold satisfying both class counts
and both lower bounds is the channel's recommendation. Only the safe channel
with greater coverage is recommended overall (confidence wins a tie). Without
both classes the status is `collecting`; adequate counts without a safe threshold
produce `no_safe_threshold`. Perfect small samples are insufficient under Wilson.

`recommended_escalation_threshold` is a **GLM review threshold**: in a future,
manually approved selective-review mode, scores below it would require GLM;
scores greater than or equal to it could adopt Jev's conclusion directly.
During training **all replies still receive GLM review**, regardless of scores.
This script does not change runtime behavior or plugin settings.

Daily statistics use the last completed UTC day. The sidecar retains recent daily
recommendations; repeated runs on one UTC day do not advance the streak. Gaps,
changed score type/threshold, or fewer than `--min-daily-samples` (default 50)
reset continuity. After `--stable-days` (default 7) consecutive completed daily
observations with the same safe recommendation and sufficient new samples,
`ready_to_stop_training=true` only reports data stability. Whether to stop training
or adopt selective review must be decided manually; GLM is never removed by the
calibrator. Keep the same output/sidecar and calibration parameters for comparable
consecutive reports.
