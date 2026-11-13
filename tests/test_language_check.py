"""Language checker contracts through real plugin discovery and backend settings."""
import json
import sys
from pathlib import Path
import httpx
from unittest.mock import Mock

import pytest

from hermes_cli.plugins import PluginManager


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text('plugins:\n  enabled: [language_check]\n')
    (tmp_path / "language-discipline.md").write_text("1. 不要使用夸张语气。")
    monkeypatch.delenv("LANGUAGE_CHECK_PG_HOST", raising=False)
    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["language_check"]
    assert loaded.enabled, loaded.error
    callback = manager._hooks["transform_llm_output"][0]
    import sys
    module = sys.modules[callback.func.__module__]
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    llm = Mock()
    monkeypatch.setattr(module.httpx, "post", llm)
    return tmp_path, manager, module, llm


def response(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text,
                                                           "reasoning_content": "ignored"}}]},
                          request=httpx.Request("POST", "https://example.com"))


def verdict(violation=False):
    return response(json.dumps(dict(violation=violation, violated_rules=["夸张"] if violation else [],
                                    analysis="检查完成", severity="low")))


def invoke(plugin):
    _, manager, _, _ = plugin
    callback = manager._hooks["transform_llm_output"][0]
    return callback(response_text="这是需要检查的原始回复，包含事实、表格与链接，必须完整保留。",
                    session_id="session", platform="cli", turn_id="turn", model="main")


def event(plugin):
    return json.loads((plugin[0] / "logs/language-check.jsonl").read_text().splitlines()[-1])


def test_pass_unchanged(plugin):
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    assert event(plugin)["violation"] is False
    assert event(plugin)["py"] == sys.executable
    assert event(plugin)["pg_error"] is None
    assert event(plugin)["key_source"] == "environ"
    assert plugin[3].call_args.kwargs["json"]["thinking"] == {"type": "disabled"}
    assert plugin[3].call_args.kwargs["timeout"] == 12
    assert plugin[3].call_args.args == ("https://api.z.ai/api/coding/paas/v4/chat/completions",)
    assert plugin[3].call_args.kwargs["json"]["max_tokens"] == 512


def test_violation_rewrites_once(plugin):
    rewritten = "事实\n\n|表|\n|---|\n|内容|\n\n```py\nx = 1\n```\n[链接](https://example.com)"
    plugin[3].side_effect = [verdict(True), response(rewritten)]
    assert invoke(plugin) == rewritten
    assert plugin[3].call_count == 2
    assert all(call.kwargs["json"]["thinking"] == {"type": "disabled"}
               for call in plugin[3].call_args_list)
    record = event(plugin)
    assert record["rewritten_text"] == rewritten
    assert record["recheck_skipped"] is True
    assert record["judge_ms"] >= 0 and record["rewrite_ms"] >= 0
    kwargs = plugin[3].call_args.kwargs
    assert kwargs["timeout"] == 8
    assert kwargs["json"]["max_tokens"] == 4096
    data = json.loads(kwargs["json"]["messages"][1]["content"])
    assert data["rules"] == "1. 不要使用夸张语气。"
    assert data["reply"] == record["reply_text"]


@pytest.mark.parametrize("error", [RuntimeError("judge error"), TimeoutError("judge timeout")])
def test_judge_failure_passes_original(plugin, error):
    plugin[3].side_effect = error
    assert invoke(plugin) is None
    assert plugin[3].call_count == 1
    assert type(error).__name__ in event(plugin)["error"]
    assert event(plugin)["judge_ms"] > 0
    assert event(plugin)["rules_chars"] == len("1. 不要使用夸张语气。")


@pytest.mark.parametrize("settings, reason", [("enabled: false", "disabled"), ("min_chars: 1000", "min_chars")])
def test_live_backend_settings_skip(plugin, settings, reason):
    plugin[0].joinpath("config.yaml").write_text(
        "plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n      settings:\n        " + settings + "\n")
    assert invoke(plugin) is None
    plugin[3].assert_not_called()
    assert event(plugin)["skipped"] == reason


def test_internal_failure_does_not_raise(plugin, monkeypatch):
    monkeypatch.setattr(plugin[2], "_check", Mock(side_effect=ValueError("internal")))
    assert invoke(plugin) is None
    assert "internal" in event(plugin)["error"]


def test_rewrite_failure_passes_original(plugin):
    plugin[3].side_effect = [verdict(True), TimeoutError("rewrite timeout")]
    assert invoke(plugin) is None
    record = event(plugin)
    assert record["violation"] is True and record["rewritten"] is False
    assert record["rewrite_ms"] > 0


def test_log_failure_does_not_lose_rewrite(plugin, monkeypatch):
    plugin[3].side_effect = [verdict(True), response("改写结果")]
    original_open = Path.open

    def open_without_log(path, *args, **kwargs):
        if path.name == "language-check.jsonl":
            raise OSError("log unavailable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_without_log)
    assert invoke(plugin) == "改写结果"


def test_held_dispatch_failure_is_fail_open(plugin):
    plugin[3].side_effect = TimeoutError("judge timeout")
    manager = plugin[1]
    rewritten, failed = manager.run_held_transforms(
        tuple(manager._hooks["transform_llm_output"]),
        response_text="原文需要足够长度以触发校验，请保留全部事实内容与格式。",
        session_id="session", platform="cli", turn_id="turn", model="main")
    assert rewritten is None
    assert failed is False


@pytest.mark.parametrize("stage", ["judge", "rewrite"])
def test_hard_wall_budget(plugin, stage):
    from threading import Event
    from time import monotonic

    release, finished = Event(), Event()

    def slow_call(*args, **kwargs):
        if stage == "rewrite" and "严格 JSON" in kwargs["json"]["messages"][0]["content"]:
            return verdict(True)
        try:
            release.wait(10)
            return verdict()
        finally:
            finished.set()

    plugin[0].joinpath("config.yaml").write_text(
        "plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n"
        "      settings:\n        judge_timeout_ms: 150\n        rewrite_timeout_s: 0.15\n")
    plugin[3].side_effect = slow_call
    started = monotonic()
    try:
        assert invoke(plugin) is None
        elapsed = monotonic() - started
        assert 0.12 <= elapsed < 2
        record = event(plugin)
        assert "TimeoutError" in record["error"]
        assert record[stage + "_ms"] >= 120
        assert record["wall_ms"] >= record[stage + "_ms"]
        assert record["rewritten"] is False
    finally:
        release.set()
        assert finished.wait(2)


def test_pg_connect_failure_keeps_reply_and_jsonl(plugin, monkeypatch):
    import psycopg2
    monkeypatch.setenv("LANGUAGE_CHECK_PG_HOST", "192.0.2.1")
    connect = Mock(side_effect=psycopg2.OperationalError("unreachable"))
    monkeypatch.setattr(psycopg2, "connect", connect)
    plugin[3].side_effect = [verdict(True), response("改写结果")]
    assert invoke(plugin) == "改写结果"
    assert event(plugin)["rewritten_text"] == "改写结果"
    assert event(plugin)["pg_error"] == "OperationalError: unreachable"
    assert connect.call_count == 1
    assert connect.call_args.kwargs["connect_timeout"] == 3


def test_pg_insert_commit_and_reconnect(plugin, monkeypatch):
    import psycopg2
    from unittest.mock import MagicMock
    monkeypatch.setenv("LANGUAGE_CHECK_PG_HOST", "127.0.0.1")
    monkeypatch.setenv("LANGUAGE_CHECK_PG_PASSWORD", "p%#!")
    first, second = MagicMock(closed=False), MagicMock(closed=False)
    connect = Mock(side_effect=[first, second])
    monkeypatch.setattr(psycopg2, "connect", connect)
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    cursor = first.cursor.return_value.__enter__.return_value
    sql, values = cursor.execute.call_args.args
    assert sql.startswith("INSERT INTO language_check_events")
    assert "session" in values and "main" in values
    columns = sql.split("(", 1)[1].split(")", 1)[0].split(", ")
    assert dict(zip(columns, values))["rules_chars"] == len("1. 不要使用夸张语气。")
    assert first.commit.call_count == 2  # schema and event
    assert connect.call_args.args == ()
    assert connect.call_args.kwargs["password"] == "p%#!"
    cursor.execute.side_effect = psycopg2.OperationalError("connection lost")
    assert invoke(plugin) is None
    first.close.assert_called_once()
    assert connect.call_count == 1  # no retry in failing event
    assert event(plugin)["violation"] is False
    assert event(plugin)["pg_error"] == "OperationalError: connection lost"
    assert invoke(plugin) is None
    assert event(plugin)["pg_error"] is None
    assert connect.call_count == 2
    assert second.commit.call_count == 2
    plugin[2]._store._close()


def test_fenced_judge_json(plugin):
    content = verdict().json()["choices"][0]["message"]["content"]
    plugin[3].return_value = response("```json\n" + content + "\n```")
    assert invoke(plugin) is None
    assert event(plugin)["violation"] is False
    assert event(plugin)["error"] is None


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_content_raises_and_fails_open(plugin, content):
    plugin[3].return_value = response(content)
    with pytest.raises(ValueError, match="Empty language check response"):
        plugin[2]._call("zai/glm-5.3-flash", 8, "judge", {})
    assert invoke(plugin) is None
    assert "ValueError" in event(plugin)["error"]


def test_http_429_raises_and_fails_open(plugin):
    plugin[3].return_value = httpx.Response(
        429, json={"error": "Insufficient balance"},
        request=httpx.Request("POST", "https://example.com"))
    with pytest.raises(httpx.HTTPStatusError):
        plugin[2]._call("zai/glm-5.3-flash", 8, "judge", {})
    plugin[3].reset_mock()
    assert invoke(plugin) is None
    assert plugin[3].call_count == 1
    assert "HTTPStatusError" in event(plugin)["error"]


def test_key_fallback_and_base_url(plugin, monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY")
    monkeypatch.setenv("LANGUAGE_CHECK_LLM_BASE_URL", "https://example.com/v4/")
    secret = Mock(return_value="fallback-key")
    monkeypatch.setattr(plugin[2], "get_secret", secret)
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    assert event(plugin)["key_source"] == "secret"
    secret.assert_called_once_with("ZAI_API_KEY")
    assert plugin[3].call_args.args == ("https://example.com/v4/chat/completions",)
    assert plugin[3].call_args.kwargs["headers"]["Authorization"] == "Bearer fallback-key"
    assert event(plugin)["error"] is None


def test_unscoped_key_failure_passes_original(plugin, monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY")
    monkeypatch.setattr(plugin[2], "get_secret", Mock(side_effect=RuntimeError("unscoped")))
    assert invoke(plugin) is None
    plugin[3].assert_not_called()
    assert "Missing ZAI_API_KEY" in event(plugin)["error"]
    assert event(plugin)["key_source"] == "missing"


def test_pg_environment_precedes_unscoped_secrets(plugin, monkeypatch):
    import sys
    import psycopg2
    from unittest.mock import MagicMock

    pg = sys.modules[plugin[2].EventStore.__module__]
    secret = Mock(side_effect=RuntimeError("unscoped"))
    monkeypatch.setattr(pg, "get_secret", secret)
    params = dict(host="127.0.0.1", port="5432", dbname="events", user="checker", password="p%#!")
    for key, suffix in [("host", "HOST"), ("port", "PORT"), ("dbname", "DB"),
                        ("user", "USER"), ("password", "PASSWORD")]:
        monkeypatch.setenv("LANGUAGE_CHECK_PG_" + suffix, params[key])
    connection = MagicMock(closed=False)
    connect = Mock(return_value=connection)
    monkeypatch.setattr(psycopg2, "connect", connect)
    store = pg.EventStore()
    try:
        store.write({"session_id": "env-session"})
        assert all(connect.call_args.kwargs[key] == value for key, value in params.items())
        secret.assert_not_called()
        assert connection.commit.call_count == 2
    finally:
        store._close()


def test_pg_unscoped_missing_parameters_are_optional(plugin, monkeypatch):
    import sys
    pg = sys.modules[plugin[2].EventStore.__module__]
    monkeypatch.delenv("LANGUAGE_CHECK_PG_HOST", raising=False)
    monkeypatch.setattr(pg, "get_secret", Mock(side_effect=RuntimeError("unscoped")))
    assert pg._parameter("HOST") is None


def test_pg_import_failure_reaches_jsonl(plugin, monkeypatch):
    monkeypatch.setenv("LANGUAGE_CHECK_PG_HOST", "127.0.0.1")
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    assert event(plugin)["pg_error"] == (
        "ModuleNotFoundError: import of psycopg2 halted; None in sys.modules")


def test_judge_rules_slice_and_soul_section(plugin):
    plugin[0].joinpath("language-discipline.md").write_text(
        "# 语言纪律\n前言\n1. 不要夸张。\n补充说明\n11. 保留事实。\n"
        "## Bad Case / 案例\n不应进入判官的案例正文\n1. 编号案例也应丢弃。\n", encoding="utf-8")
    plugin[0].joinpath("SOUL.md").write_text(
        "# 身份\n不相关身份文本\n### 自动提炼\n1. 简洁表达。\n避免空话。\n"
        "### 其他\n不相关尾部\n", encoding="utf-8")
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    data = json.loads(plugin[3].call_args.kwargs["json"]["messages"][1]["content"])
    assert data["rules"] == "1. 不要夸张。\n11. 保留事实。\n\n### 自动提炼\n1. 简洁表达。\n避免空话。"
    assert event(plugin)["rules_chars"] == len(data["rules"])


def test_judge_timeout_ms_setting(plugin):
    plugin[0].joinpath("config.yaml").write_text(
        "plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n"
        "      settings:\n        judge_timeout_ms: 2500\n")
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    assert plugin[3].call_args.kwargs["timeout"] == 2.5


def pre_call(plugin, messages):
    return plugin[1].invoke_hook('pre_llm_call', conversation_history=messages,
                                 user_message='问题', session_id='same-session')


def test_dynamic_context_each_turn_and_request_copy(plugin):
    from agent.turn_context import compose_user_api_content

    rules = '- 禁套话：“反例”→改：直说'
    plugin[0].joinpath('language-rules-dynamic.md').write_text(rules)
    messages = [{'role': 'system', 'content': '固定提示'},
                {'role': 'user', 'content': '问题'}]
    for _ in range(2):
        results = pre_call(plugin, messages)
        assert results == [{'context': '【语言纪律-动态】回复时严格遵守：\n' + rules}]
        request = list(messages)
        request[-1] = dict(request[-1])
        request[-1]['content'] = compose_user_api_content(
            request[-1]['content'], '', results[0]['context'])
        assert request[-1]['content'] == '问题\n\n' + results[0]['context']
        assert messages[-1] == {'role': 'user', 'content': '问题'}
        assert request[0] is messages[0]
    assert event(plugin)['hook'] == 'pre_llm_call'
    assert event(plugin)['injected_chars'] == len(rules)
    plugin[3].assert_not_called()


@pytest.mark.parametrize('content', [None, '', '  \n'])
def test_dynamic_empty_or_missing(plugin, content):
    if content is not None:
        plugin[0].joinpath('language-rules-dynamic.md').write_text(content)
    assert pre_call(plugin, []) == []
    assert event(plugin)['injected_chars'] == 0


def test_dynamic_limit_and_mtime_cache(plugin, monkeypatch):
    path = plugin[0] / 'language-rules-dynamic.md'
    path.write_text('规' * 7000)
    assert len(pre_call(plugin, [])[0]['context'].split('\n', 1)[1]) == 6400
    assert event(plugin)['injected_chars'] == 6400
    original = Path.read_text
    def cached_read(self, *args, **kwargs):
        if self == path:
            raise AssertionError('unchanged file must be cached')
        return original(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_text', cached_read)
        assert pre_call(plugin, [])
    path.write_text('新规则')
    import os
    stamp = path.stat().st_mtime_ns
    os.utime(path, ns=(stamp, stamp + 1000000))
    assert pre_call(plugin, [])[0]['context'].endswith('新规则')
    path.unlink()
    assert pre_call(plugin, []) == []


def test_dynamic_failure_returns_unchanged(plugin, monkeypatch):
    messages = [{'role': 'user', 'content': '问题'}]
    monkeypatch.setattr(plugin[2], '_dynamic_rules', Mock(side_effect=OSError('read failed')))
    assert pre_call(plugin, messages) == []
    assert messages == [{'role': 'user', 'content': '问题'}]
    assert event(plugin)['injected_chars'] == 0


@pytest.mark.parametrize('text,limit,expected', [
    ('old\nnew\nlast', 8, 'new\nlast'),
    ('old\nnew\nlast', 7, 'last'),
    ('old\n' + '新' * 20, 10, '新' * 10),
    ('old\nnew\n', 4, 'new\n'),
])
def test_dynamic_whole_line_budget(plugin, text, limit, expected):
    plugin[0].joinpath('config.yaml').write_text(
        'plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n'
        f'      settings:\n        inject_max_chars: {limit}\n')
    plugin[0].joinpath('language-rules-dynamic.md').write_text(text)
    assert pre_call(plugin, [])[0]['context'].split('\n', 1)[1] == expected
    assert event(plugin)['file_chars'] == len(text)
    assert event(plugin)['truncated'] is True


def test_dynamic_warning_deduplicates_by_mtime(plugin, caplog):
    import os
    path = plugin[0] / 'language-rules-dynamic.md'
    path.write_text('old\n' + '新' * 6500)
    for _ in range(3):
        pre_call(plugin, [])
    warnings = [r.message for r in caplog.records if 'dropped_lines=' in r.message]
    assert warnings == ['Dynamic language rules truncated: file_chars=6504 '
                        'window_tokens=128000 budget_chars=6400 dropped_lines=1']
    stamp = path.stat().st_mtime_ns
    os.utime(path, ns=(stamp, stamp + 1000000))
    pre_call(plugin, [])
    assert sum('dropped_lines=' in r.message for r in caplog.records) == 2


@pytest.mark.parametrize('rules,warning', [
    ('- short\n', False),
    ('新' * 6401, True),
    ('# heading\n' + '- x\n' * 30, True),
    ('# heading\n' + '- x\n' * 29, False),
])
def test_post_compaction_budget(plugin, caplog, rules, warning):
    plugin[0].joinpath('language-rules-dynamic.md').write_text(rules)
    plugin[1].invoke_hook('post_compaction', session_id='s', old_session_id='',
                          kind='summary', reason='manual', before_tokens=2000,
                          after_tokens=500, messages_removed=5)
    assert any('dynamic-rules-over-budget' in r.message for r in caplog.records) is warning
    log = plugin[0] / 'logs/language-check.jsonl'
    if warning:
        record = event(plugin)
        assert record == dict(hook='post_compaction', ts=record['ts'],
                              file_chars=len(rules), rule_lines=rules.count('- '),
                              warning='dynamic-rules-over-budget')
    else:
        assert not log.exists()


@pytest.mark.parametrize('settings,window,budget', [
    ('', 128000, 6400),
    ('context_window_tokens: 20000', 20000, 1000),
    ('context_window_tokens: 20000\n        inject_budget_percent: 2.5', 20000, 500),
    ('inject_max_chars: 100', 128000, 100),
    ('inject_max_chars: 9000', 128000, 6400),
    ('inject_max_chars: 0', 128000, 6400),
    ('inject_budget_percent: 0', 128000, 0),
])
def test_injection_percentage_budget_without_payload_window(plugin, settings, window, budget):
    plugin[0].joinpath('config.yaml').write_text(
        'plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n'
        f'      settings:\n        {settings}\n')
    plugin[0].joinpath('language-rules-dynamic.md').write_text('规' * 7000)
    result = pre_call(plugin, [])
    if budget:
        assert result[0]['context'].split('\n', 1)[1] == '规' * budget
    else:
        assert result == []
    record = event(plugin)
    assert record['window_tokens'] == window
    assert record['budget_chars'] == budget
    assert record['injected_chars'] == budget
    assert record['file_chars'] == 7000
    assert record['truncated'] is True


@pytest.mark.parametrize('settings,warning', [
    ('context_window_tokens: 1000', True),
    ('context_window_tokens: 10000', False),
    ('inject_max_chars: 50', True),
    ('context_window_tokens: 1000\n        inject_budget_percent: 20', False),
])
def test_compaction_uses_percentage_and_hard_cap(plugin, caplog, settings, warning):
    plugin[0].joinpath('config.yaml').write_text(
        'plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n'
        f'      settings:\n        {settings}\n')
    plugin[0].joinpath('language-rules-dynamic.md').write_text('规' * 100)
    plugin[1].invoke_hook('post_compaction', session_id='s')
    assert any('dynamic-rules-over-budget' in r.message for r in caplog.records) is warning
