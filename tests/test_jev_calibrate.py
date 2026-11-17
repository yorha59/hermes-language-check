"""Calibration contracts using synthetic truth and daily observations."""
from datetime import UTC, datetime, timedelta
import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from scripts import jev_calibrate as jc

NOW = datetime(2026, 11, 13, tzinfo=UTC)


def rows(n=400, confidence=.9, probability=.8, correct=True, day=None):
    return [dict(id=i, ts=day or NOW-timedelta(days=1), conclusion=bool(i % 2),
                 violation=bool(i % 2) if correct else not bool(i % 2),
                 confidence=confidence, overall_probability=probability) for i in range(n)]


def test_independent_channels_lowest_safe_threshold():
    data = rows(800) + rows(100, .2, .5, False)
    report, _ = jc.calibrate(data, now=NOW)
    assert report['channels']['confidence']['recommended_escalation_threshold'] == .21
    assert report['channels']['overall_probability']['recommended_escalation_threshold'] == .51
    assert report['recommended_score_type'] == 'confidence'
    assert len(report['channels']['confidence']['thresholds']) == 100
    # Probability can retain correct answers that confidence must exclude.
    data += rows(100, .1, .8)
    report, _ = jc.calibrate(data, now=NOW)
    assert report['recommended_score_type'] == 'overall_probability'
    assert report['recommended_escalation_threshold'] == .51
    assert report['trusted_coverage'] == .9


@pytest.mark.parametrize('data,status', [
    (rows(20), 'collecting'),
    ([dict(r, conclusion=False, violation=False) for r in rows(1000)], 'collecting'),
    (rows(800, correct=False), 'no_safe_threshold'),
    (rows(100), 'no_safe_threshold'),
])
def test_no_unsafe_recommendation(data, status):
    report, _ = jc.calibrate(data, now=NOW)
    assert report['status'] == status
    assert report['recommended_escalation_threshold'] is None
    assert not report['stability']['ready_to_stop_training']


def test_wilson_small_perfect_sample_is_not_safe():
    assert jc.wilson(50, 50) < .98
    assert jc.wilson(200, 200) >= .98
    assert jc.wilson(0, 0) is None
    m = jc.metrics(rows(4), 'confidence', 0, 4)
    assert (m['tp'], m['fp'], m['tn'], m['fn']) == (2, 0, 2, 0)
    assert m['agreement'] == 1
    assert m['violation_precision'] == m['ok_precision'] == 1


def test_daily_seven_completed_days_and_repeated_run():
    state = {}
    data = []
    for i in range(7):
        now = NOW + timedelta(days=i)
        data += rows(400, day=now-timedelta(days=1))
        report, state = jc.calibrate(data, state=state, now=now)
        assert report['daily']['new_samples'] == 400
        assert report['stability']['stable_days'] == i+1
        assert report['stability']['ready_to_stop_training'] is (i == 6)
        repeat, state = jc.calibrate(data, state=state, now=now)
        assert repeat['stability'] == report['stability']
    report, state = jc.calibrate(data, state=state, now=NOW+timedelta(days=7))
    assert report['stability']['stable_days'] == 0
    assert not report['stability']['ready_to_stop_training']


def test_change_or_low_sample_day_resets_stability():
    report, state = jc.calibrate(rows(400), now=NOW)
    data = rows(400) + rows(400, day=NOW) + rows(200, .1, .1, False, NOW)
    report, state = jc.calibrate(data, state=state, now=NOW+timedelta(days=1))
    assert report['recommended_escalation_threshold'] == .11
    assert report['stability']['stable_days'] == 1
    data += rows(20, day=NOW+timedelta(days=1))
    report, _ = jc.calibrate(data, state=state, now=NOW+timedelta(days=2))
    assert report['stability']['stable_days'] == 0


def test_missing_day_breaks_streak():
    _, state = jc.calibrate(rows(), now=NOW)
    report, _ = jc.calibrate(rows()+rows(day=NOW+timedelta(days=1)),
                             now=NOW+timedelta(days=2), state=state)
    assert report['stability']['stable_days'] == 1


def test_judge_label_follows_config(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'plugins:\n  entries:\n    language_check:\n      settings:\n'
        '        judge_model: zai/glm-5.3\n'
    )
    monkeypatch.setattr(jc, 'get_hermes_home', lambda: home)
    assert jc.judge_model() == 'zai/glm-5.3'
    assert jc.judge_label() == 'glm-5.3'
    report, _ = jc.calibrate(rows(n=20), now=NOW)
    text = jc.summary_text(report, rows(n=20))
    assert text.startswith('Jev/glm-5.3 校准日报')
    assert 'glm-5.3违规:' in text
    assert 'Flash违规' not in text


@pytest.mark.parametrize('config,expected', [
    ('language_check:\n  settings:\n    judge_model: provider/model\n', 'provider/model'),
    ('judge_model: outside\nlanguage_check:\n  enabled: true\n'
     'other:\n  judge_model: sibling\n', ''),
    ('plugins:\n  language_check:\n    enabled: true\n'
     '  other:\n    judge_model: sibling\n', ''),
    ('other:\n  judge_model: outside\n', ''),
    ('language_check:\n  # comment\n\n  judge_model: "provider/model" # note\n', 'provider/model'),
    ("language_check: # settings\n  judge_model: 'provider/model' # note\n", 'provider/model'),
    ('language_check:\n  judge_model: "provider/model#tag" # note\n', 'provider/model#tag'),
    ('language_check:\n  judge_model: first # note\n  judge_model: second\n', 'first'),
])
def test_judge_model_block_scope_and_values(tmp_path, config, expected):
    (tmp_path / 'config.yaml').write_text(config)
    assert jc.judge_model(tmp_path) == expected


def test_judge_model_missing_config(tmp_path):
    assert jc.judge_model(tmp_path) == ''


def test_judge_model_large_missing_value_finishes(tmp_path):
    # The old multiline wildcard could repartition these lines exponentially.
    (tmp_path / 'config.yaml').write_text(
        'language_check:\n' + '  enabled: true\n' * 100_000
        + 'other:\n  judge_model: outside\n'
    )
    result = subprocess.run(
        [sys.executable, '-c',
         'from scripts.jev_calibrate import judge_model; '
         'import sys; assert judge_model(sys.argv[1]) == ""', str(tmp_path)],
        capture_output=True, text=True, timeout=2, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dry_run_and_atomic_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jc, 'read_events', lambda: rows())
    output = tmp_path/'nested'/'calibration.json'
    assert jc.main(['--dry-run', '--output', str(output)]) == 0
    assert not output.parent.exists()
    out = capsys.readouterr().out
    assert out.startswith('Jev/') and '校准日报' in out.splitlines()[0]
    assert 'Flash违规' not in out
    assert '样本: 400' in out
    assert '推荐复审阈值:' in out
    assert '真值模型:' in out
    replace = jc.os.replace
    calls = []
    def record(src, dest):
        assert json.loads(jc.Path(src).read_text())['schema_version'] == 1
        calls.append(dest)
        replace(src, dest)
    monkeypatch.setattr(jc.os, 'replace', record)
    assert jc.main(['--output', str(output)]) == 0
    assert json.loads(output.read_text())['sample_count'] == 400
    assert calls == [output, output.with_name(output.name+'.state.json')]
    assert len(list(output.parent.iterdir())) == 2


def test_readonly_pg_selects_no_reply_or_credentials(monkeypatch):
    import psycopg2
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [(1, NOW, False, True, .5, .6)]
    monkeypatch.setattr(jc, '_parameter', lambda suffix: 'localhost' if suffix=='HOST' else None)
    monkeypatch.setattr(psycopg2, 'connect', lambda **kwargs: connection)
    result = jc.read_events()
    connection.set_session.assert_called_once_with(readonly=True, autocommit=False)
    assert result[0]['violation'] is True
    sql = cursor.execute.call_args.args[0]
    assert 'reply_text' not in sql and 'password' not in sql
    assert "jev_training_status = 'trained'" in sql
    connection.rollback.assert_called_once()
    connection.close.assert_called_once()


def test_pg_failure_redacts_details(tmp_path, monkeypatch, capsys):
    def fail():
        raise RuntimeError('password=private')
    monkeypatch.setattr(jc, 'read_events', fail)
    assert jc.main(['--dry-run', '--output', str(tmp_path/'out')]) == 1
    assert capsys.readouterr().out == 'status=pg_unavailable error=RuntimeError\n'


def test_atomic_replace_failure_preserves_existing_file(tmp_path, monkeypatch):
    folder = tmp_path/'atomic'
    folder.mkdir()
    output = folder/'out.json'
    output.write_text('{"previous": true}')
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(jc.os, 'replace', fail)
    with pytest.raises(OSError):
        jc.atomic_write(output, {'schema_version': 1})
    assert json.loads(output.read_text()) == {'previous': True}
    assert list(folder.iterdir()) == [output]


def test_daily_metrics_use_completed_day_only():
    old = rows(800, day=NOW-timedelta(days=2))
    new = rows(100, correct=False)
    report, _ = jc.calibrate(old+new, now=NOW, target=.7)
    assert report['daily']['new_samples'] == 100
    assert report['daily']['agreement'] == 0
    assert report['agreement'] > .8


@pytest.mark.parametrize('host', [None, ''])
def test_jsonl_calibration_without_pg_import(tmp_path, monkeypatch, capsys, host):
    import builtins
    import runpy

    monkeypatch.setattr(jc, '_parameter', lambda suffix: host if suffix == 'HOST' else None)
    monkeypatch.setattr(jc, 'get_hermes_home', lambda: tmp_path)
    from scripts import language_rules_distill
    monkeypatch.setattr(language_rules_distill, 'get_hermes_home', lambda: tmp_path)
    monkeypatch.delenv('LANGUAGE_CHECK_LOG_PATH', raising=False)
    original_import = builtins.__import__

    def no_pg(name, *args, **kwargs):
        if name == 'psycopg2' or name.startswith('psycopg2.'):
            pytest.fail('JSONL mode imported psycopg2')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_pg)
    runpy.run_path(jc.__file__, run_name='calibrate_import_probe')
    assert jc.main(['--dry-run']) == 0
    assert '样本: 0' in capsys.readouterr().out
    log = tmp_path / 'logs' / 'language-check.jsonl'
    log.parent.mkdir()
    event = dict(hook='jev_training', jev_training_status='trained', ts=NOW.isoformat(),
                 violation=True, jev_conclusion=True, jev_overall_confidence=.9,
                 jev_overall_probability=.8)
    invalid = [dict(event, hook='jev_shadow'), dict(event, jev_training_status='pending'),
               dict(event, violation=None), dict(event, jev_conclusion=None),
               dict(event, jev_overall_probability=None), dict(event, ts='bad'),
               dict(event, jev_overall_confidence=2), dict(event, jev_overall_probability='bad')]
    log.write_text('broken\n[]\n' + '\n'.join(json.dumps(e) for e in invalid + [event]) + '\n')
    assert jc.main(['--dry-run']) == 0
    out = capsys.readouterr().out
    assert '样本: 1' in out and 'collecting' in out
    assert '"schema_version"' not in out
    monkeypatch.setenv('LANGUAGE_CHECK_LOG_PATH', str(log))
    assert len(jc.read_events()) == 1
    assert jc.main(['--dry-run', '--log-path', str(log), '--json']) == 0
    assert '"sample_count": 1' in capsys.readouterr().out
