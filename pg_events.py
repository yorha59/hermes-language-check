"""Optional single-connection PostgreSQL event sink; JSONL remains independent."""
import logging
import os
from threading import Lock
from time import monotonic

from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)
COLUMNS = (
    'ts', 'session_id', 'platform', 'turn_id', 'violation', 'violated_rules',
    'analysis', 'severity', 'rewritten', 'recheck_skipped', 'reply_chars',
    'judge_ms', 'rewrite_ms', 'wall_ms', 'error', 'skipped', 'reply_text',
    'rewritten_text', 'model', 'py', 'pg_error', 'key_source', 'rules_chars',
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS language_check_events (
 id bigserial PRIMARY KEY, ts timestamptz NOT NULL,
 session_id text, platform text, turn_id text,
 violation boolean, violated_rules text[], analysis text, severity text,
 rewritten boolean, recheck_skipped boolean,
 reply_chars int, judge_ms real, rewrite_ms real, wall_ms real,
 error text, skipped text, reply_text text, rewritten_text text, model text
);
ALTER TABLE language_check_events ADD COLUMN IF NOT EXISTS py text;
ALTER TABLE language_check_events ADD COLUMN IF NOT EXISTS pg_error text;
ALTER TABLE language_check_events ADD COLUMN IF NOT EXISTS key_source text;
ALTER TABLE language_check_events ADD COLUMN IF NOT EXISTS rules_chars int;
CREATE INDEX IF NOT EXISTS idx_lc_ts ON language_check_events (ts);
CREATE INDEX IF NOT EXISTS idx_lc_violation ON language_check_events (violation) WHERE violation;
"""


def _parameter(suffix):
    name = 'LANGUAGE_CHECK_PG_' + suffix
    value = os.environ.get(name)
    if value:
        return value
    try:
        return get_secret(name)
    except Exception:
        return None


class EventStore:
    def __init__(self):
        self._connection = None
        self._identity = None
        self._lock = Lock()

    def _close(self):
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                logger.warning('Language check PG close failed')

    def write(self, event=None, *, started=None):
        # Do not queue replies behind another hook's database operation.
        if not self._lock.acquire(blocking=False):
            return
        try:
            params = {key: _parameter(suffix) for key, suffix in
                      [('host', 'HOST'), ('port', 'PORT'), ('dbname', 'DB'),
                       ('user', 'USER'), ('password', 'PASSWORD')]}
            identity = tuple(params.values())
            if identity != self._identity:
                self._close()
                self._identity = identity
            if not params['host']:
                return
            if self._connection is None or self._connection.closed:
                import psycopg2
                # Keyword parameters preserve passwords containing %, # and !.
                self._connection = psycopg2.connect(
                    **{key: value for key, value in params.items() if value is not None},
                    connect_timeout=3, options='-c statement_timeout=1000 -c lock_timeout=1000',
                    keepalives=1, keepalives_idle=3, keepalives_interval=1, keepalives_count=1,
                    tcp_user_timeout=3000,
                )
                with self._connection.cursor() as cursor:
                    cursor.execute(SCHEMA)
                self._connection.commit()
            if event is not None:
                if started is not None:
                    event["wall_ms"] = (monotonic() - started) * 1000
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        'INSERT INTO language_check_events (' + ', '.join(COLUMNS) +
                        ') VALUES (' + ', '.join(['%s'] * len(COLUMNS)) + ')',
                        tuple(event.get(column) for column in COLUMNS),
                    )
                self._connection.commit()
        except Exception as exc:
            # Never log the exception/DSN: driver messages can contain credentials.
            logger.warning('Language check PG unavailable; skipping database event')
            self._close()
            return f"{type(exc).__name__}: {exc}"
        finally:
            self._lock.release()
