"""Contracts for the standalone language rule distiller; all I/O paths are temporary."""
import json
from unittest.mock import MagicMock

import httpx
import pytest

from scripts import language_rules_distill as distill


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ('soul', 'archive', 'state', 'dynamic')}
    for name, path in paths.items():
        monkeypatch.setenv(f'DISTILL_{name.upper()}_PATH', str(path))
    paths['soul'].write_bytes(b'prefix\r\n## Language Discipline\r\nold\r\n\r\n## Boundaries\r\ntail\r\n')
    paths['archive'].write_text('original archive\n')
    monkeypatch.setenv('ZAI_API_KEY', 'test-key')
    rows = [(i, ['禁用套话'], '分析', 'medium', '回复' * 150, '') for i in range(1, 4)]
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.side_effect = lambda: [r for r in rows if r[0] > cursor.execute.call_args.args[1][0]]
    monkeypatch.setattr(distill.psycopg2, 'connect', MagicMock(return_value=connection))
    response = MagicMock()
    response.json.return_value = {'choices': [{'message': {'content': json.dumps({
        'rule_line': '- 禁用套话：“回复”→改：直接表达结论', 'confidence': 'high'}, ensure_ascii=False)}}]}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(distill.httpx, 'post', post)
    return paths, rows, response, post


def test_below_threshold(setup_run, capsys):
    paths, _, _, post = setup_run
    original = paths['soul'].read_bytes()
    assert distill.main(['--threshold', '4']) == 0
    assert paths['soul'].read_bytes() == original
    assert paths['archive'].read_text() == 'original archive\n'
    assert json.loads(paths['state'].read_text()) == {'last_event_id': 3}
    post.assert_not_called()
    assert 'no changes' in capsys.readouterr().out


def test_high_and_idempotence(setup_run):
    paths, _, _, post = setup_run
    original_soul = paths['soul'].read_bytes()
    assert distill.main([]) == 0
    dynamic = paths['dynamic'].read_bytes()
    assert paths['soul'].read_bytes() == original_soul
    soul = paths['soul'].read_bytes()
    assert soul.startswith(b'prefix\r\n## Language Discipline\r\nold\r\n\r\n')
    assert soul.endswith(b'## Boundaries\r\ntail\r\n')
    assert dynamic.decode().startswith(distill.DYNAMIC_HEADING + '\n')
    assert '- 禁用套话：“回复”→改：直接表达结论' in dynamic.decode()
    archive = paths['archive'].read_text()
    assert archive.startswith('original archive\n')
    assert '## 自动提炼 ' in archive and '[1, 2, 3]' in archive
    assert post.call_args.kwargs['timeout'] == 30
    assert post.call_args.kwargs['json']['thinking'] == {'type': 'disabled'}
    examples = json.loads(post.call_args.kwargs['json']['messages'][1]['content'])['examples']
    assert len(examples) == 3 and len(examples[0]['reply_text']) == 200
    assert distill.main([]) == 0
    # Replay the same batch to also exercise document-based deduplication.
    paths['state'].unlink()
    assert distill.main([]) == 0
    assert paths['soul'].read_bytes() == soul
    assert paths['dynamic'].read_bytes() == dynamic
    assert paths['archive'].read_text() == archive


@pytest.mark.parametrize('content,expected', [
    ('{"rule_line":"- 禁用套话：“回复”→改：直接表达结论","confidence":"medium"}', False),
    ('```json\n{"rule_line":"- 禁用套话：“回复”→改：直接表达结论","confidence":"high"}\n```', True),
    ('not JSON', False),
])
def test_llm_results(setup_run, content, expected):
    paths, _, response, _ = setup_run
    response.json.return_value['choices'][0]['message']['content'] = content
    assert distill.main([]) == 0
    assert paths['dynamic'].exists() is expected
    assert paths['state'].exists()


def test_dry_run(setup_run, capsys):
    paths, _, _, _ = setup_run
    before = {name: p.read_bytes() if p.exists() else None for name, p in paths.items()}
    assert distill.main(['--dry-run']) == 0
    assert {name: p.read_bytes() if p.exists() else None for name, p in paths.items()} == before
    assert '直接表达结论' in capsys.readouterr().out


@pytest.mark.parametrize('failure', ['pg', 'llm', 'soul'])
def test_failures_preserve_cursor(setup_run, monkeypatch, capsys, failure):
    paths, _, _, post = setup_run
    original = paths['soul'].read_bytes()
    if failure == 'pg':
        distill.psycopg2.connect.side_effect = distill.psycopg2.OperationalError('unavailable')
    elif failure == 'llm':
        post.side_effect = httpx.ReadTimeout('timeout')
    else:
        replace = distill.os.replace

        def fail_soul(source, target):
            if target == paths['dynamic']:
                raise OSError('write failed')
            return replace(source, target)

        monkeypatch.setattr(distill.os, 'replace', fail_soul)
    assert distill.main([]) == 1
    assert paths['soul'].read_bytes() == original
    assert not paths['state'].exists()
    assert 'failed' in capsys.readouterr().err


@pytest.mark.parametrize('line', [
    '- 禁用套话：“回复”→改：' + '直' * 60,
    '- 禁用套话：“回复”直接表达结论',
    '- 禁用套话：回复→改：直接表达结论',
    '- 禁用套话：“回复”→改：直接表达结论\n',
])
def test_invalid_rule_line(setup_run, capsys, line):
    paths, _, response, _ = setup_run
    original = paths['soul'].read_bytes()
    response.json.return_value['choices'][0]['message']['content'] = json.dumps({
        'rule_line': line, 'confidence': 'high'})
    assert distill.main([]) == 0
    assert paths['soul'].read_bytes() == original
    assert json.loads(capsys.readouterr().out)['skipped']['禁用套话'] == 'invalid rule_line'


@pytest.mark.parametrize('limit', [30, 2])
def test_full_section(setup_run, capsys, limit):
    paths, _, _, _ = setup_run
    text = distill.DYNAMIC_HEADING + '\n' + '\n'.join(
        f'- 禁模式{i}：“反例”→改：直说' for i in range(limit)) + '\n'
    paths['dynamic'].write_text(text)
    args = [] if limit == 30 else ['--max-entries', str(limit)]
    assert distill.main(args) == 0
    assert paths['dynamic'].read_text() == text
    assert paths['archive'].read_text() == 'original archive\n'
    output = capsys.readouterr().out.splitlines()
    assert output[0] == (f'WARNING: dynamic rules full ({limit}/{limit}) — '
                         'oldest rules will be dropped by injection truncation')
    summary = json.loads(output[1])
    assert summary['skipped']['禁用套话'] == f'section full ({limit}/{limit})'
    assert summary['section_entries'] == limit


def test_summary_size(setup_run, capsys):
    paths, _, _, _ = setup_run
    assert distill.main([]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['soul_bytes'] == len(paths['soul'].read_bytes())
    assert summary['dynamic_bytes'] == len(paths['dynamic'].read_bytes())
    assert summary['section_entries'] == 1


def test_exact_length_and_batch_capacity(setup_run, capsys):
    paths, rows, response, _ = setup_run
    line = '- 禁用套话：“' + '回复' * 19 + '”→改：直接表达结论'
    line = line.replace('回复' * 19, '回复' * 19 + '回' * (60 - len(line)))
    assert len(line) == 60
    rows[:] = [(i, ['禁用套话', '其他模式'], *row[2:])
               for i, row in enumerate(rows, 1)]
    response.json.return_value['choices'][0]['message']['content'] = json.dumps({
        'rule_line': line, 'confidence': 'high'})
    assert distill.main(['--max-entries', '1']) == 0
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith('WARNING: dynamic rules full (1/1)')
    summary = json.loads(output[1])
    assert summary['new_rules'] == [line]
    assert summary['skipped']['其他模式'] == 'section full (1/1)'
    assert summary['section_entries'] == 1


def test_write_soul_opt_in(setup_run):
    paths, _, _, _ = setup_run
    assert distill.main([]) == 0
    assert distill.main(['--write-soul']) == 0
    soul = paths['soul'].read_bytes()
    assert distill.HEADING in soul.decode()
    assert soul.startswith(b'prefix\r\n## Language Discipline\r\nold\r\n\r\n')
    assert soul.endswith(b'## Boundaries\r\ntail\r\n')
    assert '- 禁用套话：“回复”→改：直接表达结论' in soul.decode()
    assert paths['dynamic'].exists()


def test_default_dynamic_path(setup_run, tmp_path, monkeypatch):
    paths, _, _, _ = setup_run
    monkeypatch.delenv('DISTILL_DYNAMIC_PATH')
    monkeypatch.setattr(distill, 'get_hermes_home', lambda: tmp_path)
    assert distill.main([]) == 0
    assert (tmp_path / 'language-rules-dynamic.md').exists()
    assert not paths['dynamic'].exists()


def test_default_does_not_require_soul(setup_run):
    paths, _, _, _ = setup_run
    paths['soul'].unlink()
    assert distill.main([]) == 0
    assert paths['dynamic'].exists()
    assert not paths['soul'].exists()
