"""Calibration contracts using synthetic truth and daily observations."""
from datetime import UTC, datetime, timedelta
import json
from unittest.mock import MagicMock

import pytest

try:
    from scripts import jev_calibrate as jc
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import jev_calibrate as jc

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


def test_dry_run_and_atomic_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jc, 'read_events', lambda: rows())
    output = tmp_path/'nested'/'calibration.json'
    assert jc.main(['--dry-run', '--output', str(output)]) == 0
    assert not output.parent.exists()
    out = capsys.readouterr().out
    assert out.startswith('Jev/GLM 校准日报')
    assert '样本: 400' in out
    assert '推荐复审阈值:' in out
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
