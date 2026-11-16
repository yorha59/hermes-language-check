#!/usr/bin/env python3
"""Read-only Jev/judge calibration. Never changes runtime settings."""
import argparse
from datetime import UTC, datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hermes_constants import get_hermes_home
from plugins.language_check.pg_events import _parameter
try:
    from scripts.language_rules_distill import jsonl_events
except ImportError:
    from language_rules_distill import jsonl_events

SQL = """SELECT id, ts, jev_conclusion, violation,
jev_overall_confidence, jev_overall_probability
FROM language_check_events
WHERE jev_training_status = 'trained' AND violation IS NOT NULL
AND jev_conclusion IS NOT NULL AND jev_overall_confidence IS NOT NULL
AND jev_overall_probability IS NOT NULL ORDER BY id"""


def read_jsonl_events(path=None):
    rows = []
    for line_id, event in jsonl_events(path):
        if event.get('hook') != 'jev_training' or event.get('jev_training_status') != 'trained':
            continue
        fields = ('ts', 'jev_conclusion', 'violation',
                  'jev_overall_confidence', 'jev_overall_probability')
        if any(event.get(key) is None for key in fields):
            continue
        if not all(isinstance(event[key], bool) for key in ('jev_conclusion', 'violation')):
            continue
        scores = [event[key] for key in fields[-2:]]
        if not all(type(score) in (int, float) and 0 <= score <= 1 for score in scores):
            continue
        try:
            stamp = datetime.fromisoformat(event['ts'])
        except (TypeError, ValueError):
            continue
        if stamp.tzinfo is None:
            continue
        rows.append(dict(id=line_id, ts=stamp, conclusion=event['jev_conclusion'],
                         violation=event['violation'], confidence=scores[0],
                         overall_probability=scores[1]))
    return rows


def read_events(log_path=None):
    params = {key: _parameter(suffix) for key, suffix in
              [('host', 'HOST'), ('port', 'PORT'), ('dbname', 'DB'),
               ('user', 'USER'), ('password', 'PASSWORD')]}
    if not params['host']:
        return read_jsonl_events(log_path)
    import psycopg2
    connection = psycopg2.connect(**{k: v for k, v in params.items() if v is not None},
                                  connect_timeout=5)
    try:
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor() as cursor:
            cursor.execute('SET LOCAL statement_timeout = 30000')
            cursor.execute(SQL)
            return [dict(zip(('id', 'ts', 'conclusion', 'violation', 'confidence',
                              'overall_probability'), row)) for row in cursor.fetchall()]
    finally:
        connection.rollback()
        connection.close()


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    return (p + z*z/(2*total) - z * math.sqrt(
        p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total)


def metrics(rows, score_type, threshold, total):
    trusted = [r for r in rows if r[score_type] >= threshold]
    tp = sum(r['conclusion'] and r['violation'] for r in trusted)
    fp = sum(r['conclusion'] and not r['violation'] for r in trusted)
    tn = sum(not r['conclusion'] and not r['violation'] for r in trusted)
    fn = sum(not r['conclusion'] and r['violation'] for r in trusted)
    return dict(threshold=threshold, trusted_coverage=len(trusted)/total if total else 0.0,
                tp=tp, fp=fp, tn=tn, fn=fn,
                agreement=(tp+tn)/len(trusted) if trusted else 0.0,
                violation_precision=tp/(tp+fp) if tp+fp else None,
                ok_precision=tn/(tn+fn) if tn+fn else None,
                violation_precision_lower_95=wilson(tp, tp+fp),
                ok_precision_lower_95=wilson(tn, tn+fn))


def scan(rows, score_type, minimum, target):
    table = [metrics(rows, score_type, i/100, len(rows)) for i in range(100)]
    safe = next((m for m in table if m['tp']+m['fp'] >= minimum
                 and m['tn']+m['fn'] >= minimum
                 and m['violation_precision_lower_95'] >= target
                 and m['ok_precision_lower_95'] >= target), None)
    enough = any(m['tp']+m['fp'] >= minimum and m['tn']+m['fn'] >= minimum
                 for m in table)
    return dict(status='ready' if safe else ('no_safe_threshold' if enough else 'collecting'),
                recommended_escalation_threshold=safe['threshold'] if safe else None,
                recommended_metrics=safe, thresholds=table)


def daily_stability(report, rows, state, now, stable_days, min_daily):
    today = now.date().isoformat()
    # Completed UTC days only: repeated runs cannot count partial days or advance a streak.
    history = dict(state.get('days', {}))
    yesterday = (now.date() - timedelta(days=1)).isoformat()
    counts = {}
    for row in rows:
        stamp = row['ts']
        if isinstance(stamp, str):
            stamp = datetime.fromisoformat(stamp)
        day = stamp.astimezone(UTC).date().isoformat()
        counts[day] = counts.get(day, 0) + 1
    recommendation = [report['recommended_score_type'], report['recommended_escalation_threshold']]
    history[yesterday] = dict(recommendation=recommendation, new_samples=counts.get(yesterday, 0))
    streak = 0
    day = now.date() - timedelta(days=1)
    if recommendation[1] is not None:
        while (entry := history.get(day.isoformat())):
            if entry['recommendation'] != recommendation or entry['new_samples'] < min_daily:
                break
            streak += 1
            day -= timedelta(days=1)
    daily_rows = []
    for row in rows:
        stamp = row['ts']
        if isinstance(stamp, str):
            stamp = datetime.fromisoformat(stamp)
        if stamp.astimezone(UTC).date().isoformat() == yesterday:
            daily_rows.append(row)
    daily_metrics = (metrics(daily_rows, recommendation[0], recommendation[1], len(daily_rows))
                     if recommendation[1] is not None else report)
    report['daily'] = dict(new_samples=counts.get(yesterday, 0), date=yesterday,
        **{k: daily_metrics[k] for k in ('trusted_coverage', 'agreement',
           'violation_precision_lower_95', 'ok_precision_lower_95')})
    report['stability'] = dict(stable_days=streak, required_stable_days=stable_days,
                              ready_to_stop_training=streak >= stable_days)
    cutoff = (now.date() - timedelta(days=max(stable_days+2, 32))).isoformat()
    return dict(schema_version=1, days={k: v for k, v in history.items() if cutoff <= k < today})


def calibrate(rows, *, minimum=50, target=0.98, now=None, state=None,
              stable_days=7, min_daily=50):
    rows = [r for r in rows if type(r['conclusion']) is bool and type(r['violation']) is bool
            and all(isinstance(r.get(k), (float, int)) and math.isfinite(r[k])
                    and 0 <= r[k] <= 1 for k in ('confidence', 'overall_probability'))]
    channels = {k: scan(rows, k, minimum, target) for k in ('confidence', 'overall_probability')}
    candidates = [k for k, c in channels.items() if c['recommended_metrics'] is not None]
    best = max(candidates, key=lambda k: channels[k]['recommended_metrics']['trusted_coverage']) if candidates else 'confidence'
    selected = channels[best]
    m = selected['recommended_metrics'] or dict(trusted_coverage=0.0, agreement=0.0,
                 violation_precision_lower_95=None, ok_precision_lower_95=None)
    now = now or datetime.now(UTC)
    model = judge_model()
    label = judge_label(model)
    report = dict(schema_version=1, generated_at=now.isoformat(), reference='judge_violation',
                  judge_model=model or label, judge_label=label,
                  sample_count=len(rows), recommended_score_type=best,
                  recommended_escalation_threshold=selected['recommended_escalation_threshold'],
                  status=selected['status'] if candidates else (
                      'collecting' if all(c['status']=='collecting' for c in channels.values())
                      else 'no_safe_threshold'), channels=channels,
                  higher_coverage_score_type=best if candidates else None,
                  target_precision=target, min_predicted_class_samples=minimum,
                  **{k: m[k] for k in ('trusted_coverage', 'agreement',
                    'violation_precision_lower_95', 'ok_precision_lower_95')})
    next_state = daily_stability(report, rows, state or {}, now, stable_days, min_daily)
    return report, next_state


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, encoding='utf-8',
                                         delete=False) as handle:
            name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def judge_model(home=None):
    """Current language_check judge_model from config; empty string if unresolved."""
    home = Path(home or get_hermes_home())
    path = home / 'config.yaml'
    if not path.is_file():
        return ''
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return ''
    match = re.search(
        r'(?ms)^[ \t]*language_check:[ \t]*\r?\n'
        r'(?:^[ \t]+(?!language_check:).*\r?\n)*?'
        r'^[ \t]+judge_model:[ \t]*([^#\r\n]+)',
        text,
    )
    if match:
        return match.group(1).strip().strip('"\'')
    return ''


def judge_label(model=None):
    """Short display name for reports, e.g. glm-5.3 / glm-5.3-flash."""
    raw = (model if model is not None else judge_model()) or 'judge'
    return raw.split('/', 1)[-1] or 'judge'


def summary_text(report, rows, label=None):
    n = len(rows)
    label = label or report.get('judge_label') or judge_label()
    judge_v = sum(r['violation'] for r in rows)
    jev_v = sum(r['conclusion'] for r in rows)
    tp = sum(r['conclusion'] and r['violation'] for r in rows)
    fp = sum(r['conclusion'] and not r['violation'] for r in rows)
    tn = sum(not r['conclusion'] and not r['violation'] for r in rows)
    fn = sum(not r['conclusion'] and r['violation'] for r in rows)
    agree = (tp + tn) / n if n else 0.0
    lines = [
        f'Jev/{label} 校准日报',
        f"状态: {report['status']} | 样本: {n} | 稳定天数: {report['stability']['stable_days']}/{report['stability']['required_stable_days']}",
        f"真值模型: {report.get('judge_model') or label}",
        f"{label}违规: {judge_v} | Jev违规: {jev_v} | 一致率: {agree:.1%}",
        f"混淆矩阵: TP={tp} FP={fp} TN={tn} FN={fn}",
        '',
        '置信度分层一致率:',
    ]
    for lo in (0.0, 0.5, 0.7, 0.8, 0.9):
        sub = [r for r in rows if r['confidence'] >= lo]
        if not sub:
            continue
        a = sum(r['conclusion'] == r['violation'] for r in sub) / len(sub)
        lines.append(
            f"- conf≥{lo:.1f}: n={len(sub)} 一致={a:.1%} Jev违规={sum(r['conclusion'] for r in sub)} "
            f"{label}违规={sum(r['violation'] for r in sub)}"
        )
    thr = report['recommended_escalation_threshold']
    lines += [
        '',
        f"推荐复审阈值: {report['recommended_score_type']}≥{thr}",
        f"可信覆盖率: {report['trusted_coverage']:.1%} | 可结束训练: {report['stability']['ready_to_stop_training']}",
        f"昨日新增: {report['daily']['new_samples']}",
    ]
    if thr is None:
        lines.append(f'说明: 样本不足或未达安全阈值，继续全量 {label} 复审。')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-path', type=Path, help='JSONL input (or LANGUAGE_CHECK_LOG_PATH)')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--json', action='store_true', help='also print full JSON')
    parser.add_argument('--min-predicted-class-samples', type=int, default=50)
    parser.add_argument('--target-precision', type=float, default=0.98)
    parser.add_argument('--stable-days', type=int, default=7)
    parser.add_argument('--min-daily-samples', type=int, default=50)
    args = parser.parse_args(argv)
    if (min(args.min_predicted_class_samples, args.stable_days, args.min_daily_samples) < 1
            or not 0 < args.target_precision <= 1):
        parser.error('sample/day limits must be positive; precision must be in (0, 1]')
    output = (args.output or Path(get_hermes_home()) / 'jev-calibration.json').expanduser()
    sidecar = output.with_name(output.name + '.state.json')
    state = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    try:
        rows = read_events(args.log_path) if args.log_path else read_events()
    except Exception as exc:
        # Never expose connection strings, credentials, or driver response text.
        print('status=pg_unavailable error=' + type(exc).__name__)
        return 1
    report, state = calibrate(rows, minimum=args.min_predicted_class_samples,
        target=args.target_precision, state=state, stable_days=args.stable_days,
        min_daily=args.min_daily_samples)
    print(summary_text(report, rows, label=report.get('judge_label')))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.dry_run:
        atomic_write(output, report)
        atomic_write(sidecar, state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
