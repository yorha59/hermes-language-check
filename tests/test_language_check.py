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


@pytest.mark.parametrize("outcome", ["pass", "failure", "skip"])
def test_transform_records_configured_judge_model(plugin, monkeypatch, outcome):
    judge_model = "zai/glm-5.3"
    plugin[0].joinpath("config.yaml").write_text(
        "plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n"
        f"      settings:\n        judge_model: {judge_model}\n"
        f"        enabled: {'false' if outcome == 'skip' else 'true'}\n")
    store = Mock()
    store.write.return_value = None
    monkeypatch.setattr(plugin[2], "_store", store)
    if outcome == "failure":
        plugin[3].side_effect = TimeoutError("judge timeout")
    else:
        plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    record = event(plugin)
    assert record["judge_model"] == judge_model
    assert record["model"] == "main"
    assert store.write.call_args.args[0]["judge_model"] == judge_model


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
                              warning='dynamic-rules-over-budget',
                              judge_model=plugin[2]._DEFAULTS['judge_model'])
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


@pytest.mark.parametrize("model, disabled, thinking", [
    ("GLM-test", True, True), ("gpt-test", True, False), ("glm-test", False, False),
])
def test_provider_registry_and_thinking(plugin, monkeypatch, model, disabled, thinking):
    from providers.base import ProviderProfile
    import providers
    registry = Mock(return_value=ProviderProfile(name="example", base_url="https://provider.test/v1",
                                                env_vars=("EXAMPLE_KEY",)))
    monkeypatch.setattr(providers, "get_provider_profile", registry)
    monkeypatch.setattr(plugin[2], "get_secret", Mock(return_value="provider-key"))
    plugin[3].return_value = response("ok")
    diagnostic = {}
    assert plugin[2]._call(model, 1, "check", {}, judge_provider="example",
                           disable_thinking=disabled, event=diagnostic) == "ok"
    assert diagnostic["key_source"] == "secret:EXAMPLE_KEY"
    assert ("thinking" in plugin[3].call_args.kwargs["json"]) is thinking
    assert plugin[3].call_args.args == ("https://provider.test/v1/chat/completions",)
    assert plugin[3].call_args.kwargs["headers"]["Authorization"] == "Bearer provider-key"
    registry.assert_called_once_with("example")


@pytest.mark.parametrize("credential", ["api_key: plain-key", "key_env: CUSTOM_KEY"])
def test_provider_config_for_judge_and_rewrite(plugin, monkeypatch, credential):
    import providers
    monkeypatch.setattr(providers, "get_provider_profile", Mock(return_value=None))
    monkeypatch.setattr(plugin[2], "get_secret", Mock(return_value="plain-key"))
    plugin[0].joinpath("config.yaml").write_text(
        f"providers:\n  custom-test:\n    api: https://custom.test/v1\n    {credential}\n"
        "plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n"
        "      settings:\n        judge_provider: custom-test\n        judge_model: gpt-test\n")
    plugin[3].side_effect = [verdict(True), response("改写")]
    assert invoke(plugin) == "改写"
    assert event(plugin)["key_source"] == ("inline" if credential.startswith("api_key:") else "secret:CUSTOM_KEY")
    for call in plugin[3].call_args_list:
        assert call.args == ("https://custom.test/v1/chat/completions",)
        assert call.kwargs["json"]["model"] == "gpt-test"
        assert "thinking" not in call.kwargs["json"]
        assert call.kwargs["headers"]["Authorization"] == "Bearer plain-key"


def test_no_configuration_disables_judge(plugin, monkeypatch, caplog):
    monkeypatch.delenv("ZAI_API_KEY")
    monkeypatch.setattr(plugin[2], "get_secret", Mock(return_value=None))
    assert invoke(plugin) is None
    plugin[3].assert_not_called()
    assert event(plugin)["key_source"] == "missing"
    assert event(plugin)["skipped"] == "judge_unconfigured"
    assert "judge disabled" in caplog.text


@pytest.mark.parametrize('name,base,mode', [
    ('zai', 'https://api.z.ai/api/coding/paas/v4', 'chat_completions'),
    ('kimi-coding', 'https://api.kimi.com/coding/', 'anthropic_messages'),
])
def test_custom_provider_overrides_builtin(plugin, monkeypatch, name, base, mode):
    import yaml
    plugin[0].joinpath('config.yaml').write_text(yaml.safe_dump({'custom_providers': [
        dict(name=name, base_url=base, api_key='inline-key', api_mode=mode)]}))
    assert plugin[2]._provider_connection(name) == (base, 'inline-key', 'inline', mode)


@pytest.mark.parametrize('source', ['inline', 'explicit_env', 'provider_env', 'secret'])
def test_custom_credential_chain(plugin, monkeypatch, source):
    import yaml
    for variable in ('GLM_API_KEY', 'ZAI_API_KEY', 'Z_AI_API_KEY', 'CUSTOM_JUDGE_KEY'):
        monkeypatch.delenv(variable, raising=False)
    secret = Mock(return_value='secret-key')
    monkeypatch.setattr(plugin[2], 'get_secret', secret)
    entry = dict(name='zai', base_url='https://custom.test/v4', api_key='',
                 api_key_env='CUSTOM_JUDGE_KEY')
    if source == 'inline':
        entry['api_key'] = 'inline-key'
    if source in ('inline', 'explicit_env'):
        monkeypatch.setenv('CUSTOM_JUDGE_KEY', 'explicit-key')
    if source != 'secret':
        monkeypatch.setenv('ZAI_API_KEY', 'provider-key')
    plugin[0].joinpath('config.yaml').write_text(yaml.safe_dump({'custom_providers': [entry]}))
    expected = {'inline': ('inline-key', 'inline'), 'explicit_env': ('explicit-key', 'environ:CUSTOM_JUDGE_KEY'),
                'provider_env': ('provider-key', 'environ:ZAI_API_KEY'), 'secret': ('secret-key', 'secret:CUSTOM_JUDGE_KEY')}
    assert plugin[2]._provider_connection('zai')[1:3] == expected[source]
    assert secret.call_count == (1 if source == 'secret' else 0)


@pytest.mark.parametrize('mode', ['chat_completions', 'anthropic_messages'])
def test_custom_wire_protocol(plugin, mode):
    import yaml
    plugin[0].joinpath('config.yaml').write_text(yaml.safe_dump({'custom_providers': [
        dict(name='custom-wire', base_url='https://custom.test/coding/', api_key='key', api_mode=mode)]}))
    plugin[3].return_value = (response('firstsecond') if mode == 'chat_completions' else
        httpx.Response(200, json={'content': [{'type': 'text', 'text': 'first'},
            {'type': 'thinking', 'thinking': 'ignored'}, {'type': 'text', 'text': 'second'}]}))
    diagnostic = {}
    assert plugin[2]._call('glm-test', 2, 'system prompt', {'input': '中文'},
                           max_tokens=123, judge_provider='custom-wire', event=diagnostic) == 'firstsecond'
    assert plugin[3].call_count == 1
    call = plugin[3].call_args
    body = call.kwargs['json']
    assert call.kwargs['timeout'] == 2
    assert body['model'] == 'glm-test' and body['max_tokens'] == 123
    user = {'role': 'user', 'content': json.dumps({'input': '中文'}, ensure_ascii=False)}
    if mode == 'anthropic_messages':
        assert call.args == ('https://custom.test/coding/v1/messages',)
        assert call.kwargs['headers'] == {'x-api-key': 'key', 'anthropic-version': '2023-06-01'}
        assert body == dict(model='glm-test', max_tokens=123, system='system prompt', messages=[user])
    else:
        assert call.args == ('https://custom.test/coding/chat/completions',)
        assert call.kwargs['headers'] == {'Authorization': 'Bearer key'}
        assert body['messages'] == [{'role': 'system', 'content': 'system prompt'}, user]
        assert body['thinking'] == {'type': 'disabled'}


def test_custom_miss_falls_back_builtin(plugin):
    plugin[0].joinpath('config.yaml').write_text('custom_providers:\n  - name: other\n    base_url: https://other.test\n')
    base, key, source, mode = plugin[2]._provider_connection('zai')
    assert base == 'https://api.z.ai/api/paas/v4'
    assert (key, source, mode) == ('test-key', 'environ:ZAI_API_KEY', 'chat_completions')


@pytest.fixture
def shadow(plugin, monkeypatch):
    module = plugin[2].jev_shadow
    monkeypatch.delenv('JEV_API_KEY', raising=False)
    monkeypatch.setattr(module, 'get_secret', Mock(return_value=None))
    queued = []
    real_schedule = module.schedule
    monkeypatch.setattr(module, 'schedule', lambda *args: queued.append(args))
    plugin[0].joinpath('config.yaml').write_text(
        'plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n'
        '      settings:\n        jev_shadow_enabled: true\n        jev_timeout_ms: 123\n')
    # HTTP executor is deterministic; no real worker or network is needed.
    from concurrent.futures import Future
    executor = Mock()
    executor.original_schedule = real_schedule

    def submit(function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    executor.submit.side_effect = submit
    monkeypatch.setattr(module, 'ThreadPoolExecutor', Mock(return_value=executor))
    return module, queued, executor


def jev_response(confidence=0.9):
    return httpx.Response(200, request=httpx.Request('POST', 'https://example.test'),
                          json={'answers': {'r1': dict(choice='violation', confidence=confidence,
                                probabilities={'violation': 0.95, 'ok': 0.05})}})


def test_shadow_disabled_never_schedules(plugin, monkeypatch):
    schedule = Mock()
    monkeypatch.setattr(plugin[2].jev_shadow, 'schedule', schedule)
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    schedule.assert_not_called()
    assert plugin[3].call_count == 1
    assert event(plugin)['jev_status'] is None


def test_jev_rules_ignores_non_integer_exclusions(plugin):
    rules = '8. eight\nnarrative\n1. one\n2. two\n### heading\n10. ten'
    assert plugin[2]._jev_rules(rules, [8, '1', True, 2.0, None, {}]) == (
        '1. one\n2. two\n10. ten')


@pytest.mark.parametrize('mode', ['training', 'shadow'])
@pytest.mark.parametrize('excluded', [[], [1, 8], list(range(1, 9))])
def test_jev_exclusions_leave_glm_rules_intact(plugin, monkeypatch, mode, excluded):
    rules = '\n'.join(f'{number}. rule {number}' for number in range(1, 9))
    plugin[0].joinpath('language-discipline.md').write_text(rules)
    plugin[0].joinpath('config.yaml').write_text(
        'plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n'
        f'      settings:\n        jev_{mode}_enabled: true\n'
        f'        jev_exclude_rule_numbers: {excluded}\n')
    module = plugin[2].jev_shadow
    start = Mock(return_value=None)
    schedule = Mock()
    monkeypatch.setattr(module, 'start_training', start)
    monkeypatch.setattr(module, 'schedule', schedule)
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    body = plugin[3].call_args.kwargs['json']
    assert json.loads(body['messages'][-1]['content'])['rules'] == rules
    record = event(plugin)
    assert record['violation'] is False
    assert record['error'] is None
    expected = [line for number, line in enumerate(rules.splitlines(), 1)
                if number not in excluded]
    selected = start if mode == 'training' else schedule
    if expected:
        selected.assert_called_once()
        sent_rules = selected.call_args.args[0]
        request, lines = module.request_body(sent_rules, 'reply', 'jev-1.14')
        assert lines == expected
        assert len(request['questions']) == len(expected)
    else:
        start.assert_not_called()
        schedule.assert_not_called()
        status = 'jev_training_status' if mode == 'training' else 'jev_status'
        assert record[status] == 'unconfigured'


def test_shadow_one_request_many_rules(plugin, shadow, monkeypatch):
    module, queued, executor = shadow
    monkeypatch.setenv('JEV_API_KEY', 'test-only')
    plugin[0].joinpath('language-discipline.md').write_text('1. 规则甲\n叙述\n3. 规则乙')
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    assert len(queued) == 1
    _rules, text, _config, original, _path, _log = queued[0]
    plugin[3].reset_mock()
    payload = {'answers': {f'r{i}': dict(choice='ok', confidence=0.9,
                probabilities={'violation': 0.05, 'ok': 0.95}) for i in (1, 2)}}
    plugin[3].return_value = httpx.Response(200, json=payload,
        request=httpx.Request('POST', 'https://example.test'))
    module.run(*queued[0])
    plugin[3].assert_called_once()
    call = plugin[3].call_args
    assert call.args == (module.ENDPOINT,)
    assert call.kwargs['timeout'] == 0.123
    body = call.kwargs['json']
    assert body['state'] == original['reply_text'] == text
    assert body['model'] == 'jev-1.14'
    assert list(body['questions']) == ['r1', 'r2']
    for key, rule in [('r1', '1. 规则甲'), ('r2', '3. 规则乙')]:
        question = body['questions'][key]
        assert question['type'] == 'choice'
        assert set(question['criteria']) == {'violation', 'ok'}
        assert rule in question['instructions'] and '代码块' in question['instructions']
    module.get_secret.assert_not_called()
    executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert event(plugin)['jev_disagrees_with_flash'] is False


@pytest.mark.parametrize('confidence,high', [(0.9, True), (0.75, True), (0.74, False)])
def test_shadow_threshold_and_correlated_jsonl(plugin, shadow, confidence, high, monkeypatch):
    module, queued, _ = shadow
    module.get_secret.return_value = 'test-only'
    store = Mock()
    monkeypatch.setattr(module, 'EventStore', Mock(return_value=store))
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    original = event(plugin)
    # Returning before the queued job runs proves shadow is outside the flash wait.
    assert original['jev_status'] is None and len(queued) == 1
    plugin[3].return_value = jev_response(confidence)
    module.run(*queued[0])
    record = event(plugin)
    rule = '1. 不要使用夸张语气。'
    assert record['hook'] == 'jev_shadow'
    assert record['turn_id'] == original['turn_id'] and record['ts'] == original['ts']
    assert record['jev_status'] == 'shadow_ok'
    assert record['jev_high_confidence_rules'] == ([rule] if high else [])
    assert record['jev_disagrees_with_flash'] is high
    assert record['jev_results'][rule] == dict(violation=True, confidence=confidence,
                                              probabilities={'violation': 0.95, 'ok': 0.05})
    assert record['jev_wall_ms'] >= 0
    store.write.assert_called_once_with(record, shadow=True)
    module.get_secret.assert_called_once_with('JEV_API_KEY')


@pytest.mark.parametrize('failure,status', [
    (httpx.ReadTimeout('sensitive body'), 'timeout'),
    (TimeoutError('sensitive body'), 'timeout'),
    (RuntimeError('sensitive body'), 'error'),
])
def test_shadow_failures_preserve_flash_rewrite(plugin, shadow, failure, status):
    module, queued, _ = shadow
    module.get_secret.return_value = 'test-only'
    plugin[3].side_effect = [verdict(True), response('权威改写')]
    assert invoke(plugin) == '权威改写'
    plugin[3].side_effect = failure
    module.run(*queued[0])
    record = event(plugin)
    assert record['jev_status'] == status
    assert record['jev_error'] == type(failure).__name__
    assert 'sensitive' not in json.dumps(record)
    assert record['jev_disagrees_with_flash'] is None


def test_shadow_unconfigured_and_scheduler_failure(plugin, shadow, monkeypatch):
    module, queued, _ = shadow
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    module.run(*queued[0])
    assert event(plugin)['jev_status'] == 'unconfigured'
    assert plugin[3].call_count == 1
    thread = Mock()
    thread.return_value.start.side_effect = RuntimeError('private')
    monkeypatch.setattr(module, 'Thread', thread)
    shadow[2].original_schedule(*queued[0])
    assert thread.call_args.kwargs['daemon'] is True
    thread.return_value.start.assert_called_once()


def test_shadow_pg_migration_and_update(plugin, monkeypatch):
    import psycopg2
    from unittest.mock import MagicMock
    pg = sys.modules[plugin[2].EventStore.__module__]
    monkeypatch.setenv('LANGUAGE_CHECK_PG_HOST', '127.0.0.1')
    connection = MagicMock(closed=False)
    monkeypatch.setattr(psycopg2, 'connect', Mock(return_value=connection))
    store = pg.EventStore()
    original = dict(ts='2026-11-13T00:00:00+00:00', turn_id='same-turn', violation=False)
    store.write(original)
    shadow_event = dict(original, jev_status='shadow_ok', jev_wall_ms=12,
                        jev_high_confidence_rules=['1. 规则'],
                        jev_disagrees_with_flash=True, jev_error=None)
    assert store.write(shadow_event, shadow=True) is None
    calls = connection.cursor.return_value.__enter__.return_value.execute.call_args_list
    for column in pg.SHADOW_COLUMNS:
        assert f'ADD COLUMN IF NOT EXISTS {column} ' in calls[0].args[0]
    insert, update = calls[1].args, calls[2].args
    assert insert[0].startswith('INSERT')
    assert update[0].startswith('UPDATE language_check_events SET')
    assert 'WHERE turn_id = %s AND ts = %s ORDER BY ts DESC, id DESC LIMIT 1' in update[0]
    assert update[1][-2:] == (original['turn_id'], original['ts'])
    assert json.loads(update[1][2]) == ['1. 规则']
    assert connection.commit.call_count == 3
    store._close()


def test_shadow_wall_deadline_discards_late_result(plugin, shadow):
    module, queued, executor = shadow
    module.get_secret.return_value = 'test-only'
    plugin[3].return_value = verdict()
    assert invoke(plugin) is None
    future = Mock()
    future.result.side_effect = TimeoutError('late body')
    executor.submit.side_effect = None
    executor.submit.return_value = future
    module.run(*queued[0])
    budget = future.result.call_args.kwargs['timeout']
    assert 0 <= budget <= 0.123
    assert event(plugin)['jev_status'] == 'timeout'
    assert event(plugin)['jev_error'] == 'TimeoutError'
    executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


@pytest.mark.parametrize('failure', [None, TimeoutError('private'), ValueError('private')])
def test_training_precedes_flash_without_shadow(plugin, monkeypatch, failure, training_done):
    module = plugin[2].jev_shadow
    plugin[0].joinpath('config.yaml').write_text(
        'plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n'
        '      settings:\n        jev_training_enabled: true\n        jev_shadow_enabled: true\n'
        '        min_chars: 9999\n')
    from threading import Event
    jev_started = Event()
    order = []
    def classify(*args):
        order.append('jev')
        jev_started.set()
        if failure:
            raise failure
        return {'1. rule': dict(violation=True, confidence=.9,
                               probabilities={'violation': .95, 'ok': .05})}
    monkeypatch.setattr(module, 'classify', classify)
    schedule = Mock()
    monkeypatch.setattr(module, 'schedule', schedule)
    def flash(*args, **kwargs):
        assert jev_started.wait(5)
        order.append('flash')
        return verdict(False)
    plugin[3].side_effect = flash
    assert invoke(plugin) is None
    assert training_done.wait(5)
    assert order == ['jev', 'flash']
    schedule.assert_not_called()
    record = event(plugin)
    assert record['violation'] is False
    assert record['jev_training_status'] == ('trained' if failure is None else
                                             'timeout' if isinstance(failure, TimeoutError) else 'error')
    for name in ('jev_results', 'jev_conclusion', 'jev_overall_confidence',
                 'jev_overall_probability', 'jev_training_status'):
        assert name in record
    assert 'private' not in json.dumps(record)


@pytest.mark.parametrize('choices,confidence,probability,conclusion', [
    (['violation', 'ok', 'violation'], .9, .8, True),
    (['ok', 'ok', 'ok'], .2, .3, False),
])
def test_training_aggregate(plugin, choices, confidence, probability, conclusion):
    answers = {str(i): dict(choice=c, confidence=score,
        probabilities={c: p, 'ok' if c == 'violation' else 'violation': 1-p})
        for i, (c, score, p) in enumerate(zip(choices, [.2, .99, .9], [.3, .99, .8]))}
    result = plugin[2].jev_shadow.aggregate(answers)
    assert result['jev_conclusion'] is conclusion
    assert result['jev_overall_confidence'] == confidence
    assert result['jev_overall_probability'] == probability
    assert [a['choice'] for a in result['jev_results'].values()] == choices


@pytest.mark.parametrize('answers', [{}, {'x': dict(choice='bad', confidence=.9,
    probabilities={'ok': .9, 'violation': .1})}])
def test_training_invalid_aggregation(plugin, answers):
    with pytest.raises(ValueError):
        plugin[2].jev_shadow.aggregate(answers)


def test_training_pg_columns_and_json(plugin, monkeypatch):
    import psycopg2
    from unittest.mock import MagicMock
    pg = sys.modules[plugin[2].EventStore.__module__]
    monkeypatch.setenv('LANGUAGE_CHECK_PG_HOST', '127.0.0.1')
    connection = MagicMock(closed=False)
    monkeypatch.setattr(psycopg2, 'connect', Mock(return_value=connection))
    store = pg.EventStore()
    record = dict(ts='2026-11-13T00:00:00+00:00', jev_results={'rule': {'choice': 'ok'}},
                  judge_model='zai/glm-5.3',
                  jev_conclusion=False, jev_overall_confidence=.8,
                  jev_overall_probability=.9, jev_training_status='trained')
    store.write(record)
    calls = connection.cursor.return_value.__enter__.return_value.execute.call_args_list
    assert 'judge_model' in pg.COLUMNS
    assert 'judge_model text' in calls[0].args[0]
    for column in ('judge_model',) + pg.TRAINING_COLUMNS:
        assert f'ADD COLUMN IF NOT EXISTS {column} ' in calls[0].args[0]
        value = calls[1].args[1][pg.COLUMNS.index(column)]
        assert (json.loads(value) if column == 'jev_results' else value) == record[column]
    store._close()


@pytest.mark.parametrize('flash_failure', [False, True])
def test_training_flash_alone_controls_rewrite_and_fail_open(plugin, monkeypatch, flash_failure, training_done):
    plugin[0].joinpath('config.yaml').write_text(
        'plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n'
        '      settings:\n        jev_training_enabled: true\n')
    monkeypatch.setattr(plugin[2].jev_shadow, 'classify', Mock(return_value={
        '1. rule': dict(violation=False, confidence=.99,
                       probabilities={'ok': .99, 'violation': .01})}))
    plugin[3].side_effect = (TimeoutError('flash timeout') if flash_failure else
                             [verdict(True), response('Flash改写')])
    assert invoke(plugin) == (None if flash_failure else 'Flash改写')
    assert training_done.wait(5)
    record = event(plugin)
    assert record['jev_training_status'] == 'trained'
    assert record['jev_conclusion'] is False
    assert record['violation'] is (None if flash_failure else True)


@pytest.fixture
def training_done(plugin, monkeypatch):
    from threading import Event
    done = Event()
    original = plugin[2]._log

    def log(path, record):
        original(path, record)
        if record.get('hook') == 'jev_training':
            done.set()
    monkeypatch.setattr(plugin[2], '_log', log)
    return done


@pytest.mark.parametrize('finish_during_insert', [False, True])
@pytest.mark.parametrize('failure', [None, TimeoutError('private'), ValueError('private')])
def test_training_blocked_future_and_insert_order(
        plugin, monkeypatch, training_done, failure, finish_during_insert):
    from concurrent.futures import Future, ThreadPoolExecutor
    from threading import Event
    module = plugin[2].jev_shadow
    plugin[0].joinpath('config.yaml').write_text(
        'plugins:\n  enabled: [language_check]\n  entries:\n    language_check:\n'
        '      settings:\n        jev_training_enabled: true\n        jev_shadow_enabled: true\n')
    pending = Future()
    started = Event()
    order = []

    def classify(*args):
        order.append('jev')
        started.set()
        return pending.result(timeout=10)

    def flash(*args, **kwargs):
        assert started.wait(5)
        order.append('flash')
        return verdict(True) if order.count('flash') == 1 else response('Flash改写')

    def finish():
        if failure:
            pending.set_exception(failure)
        else:
            pending.set_result({'1. rule': dict(violation=False, confidence=.99,
                                probabilities={'ok': .99, 'violation': .01})})

    def insert(record, **kwargs):
        if finish_during_insert:
            finish()
            assert not training_done.is_set()
        assert record['jev_training_status'] == 'pending'
        assert record['judge_model'] == plugin[2]._DEFAULTS['judge_model']
        assert 'hook' not in record
        json.dumps(record)
        order.append('insert')

    store = Mock()
    def update(record, **kwargs):
        assert order.index('insert') < len(order)
        order.append('update')
    store.write.side_effect = update
    monkeypatch.setattr(module, 'classify', classify)
    monkeypatch.setattr(module, 'EventStore', Mock(return_value=store))
    monkeypatch.setattr(plugin[2]._store, 'write', insert)
    shadow = Mock()
    monkeypatch.setattr(module, 'schedule', shadow)
    plugin[3].side_effect = flash
    with ThreadPoolExecutor(max_workers=1) as executor:
        transformed = executor.submit(invoke, plugin)
        try:
            assert transformed.result(timeout=5) == 'Flash改写'
            initial = json.loads(plugin[0].joinpath(
                'logs/language-check.jsonl').read_text().splitlines()[0])
            assert initial['jev_training_status'] == 'pending'
            assert pending.done() is finish_during_insert
            assert order[:4] == ['jev', 'flash', 'flash', 'insert']
            if not finish_during_insert:
                assert len(order) == 4
        finally:
            if not pending.done():
                finish()
    assert training_done.wait(5)
    final = event(plugin)
    assert final['hook'] == 'jev_training'
    assert final['turn_id'] == initial['turn_id']
    assert final['ts'] == initial['ts']
    assert final['rewritten_text'] == 'Flash改写'
    assert final['violation'] is True
    assert final['jev_training_status'] == ('trained' if failure is None else
        'timeout' if isinstance(failure, TimeoutError) else 'error')
    assert order[-1] == 'update'
    shadow.assert_not_called()
