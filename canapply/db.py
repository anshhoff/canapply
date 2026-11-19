"""Thin Postgres helpers. One pool per process; rows come back as dicts."""
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import config

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        kwargs = {"row_factory": dict_row, "autocommit": True}
        if "-pooler" in config.DATABASE_URL:          # pgbouncer in transaction mode cannot keep prepared statements
            kwargs["prepare_threshold"] = None
        # A serverless database (Neon) suspends when idle and drops our connections. check= tests a connection
        # before handing it out and replaces a dead one; max_idle stops us holding connections we do not need.
        _pool = ConnectionPool(
            config.DATABASE_URL, min_size=1, max_size=4, configure=_configure, kwargs=kwargs,
            check=ConnectionPool.check_connection, max_idle=120, open=True,
        )
    return _pool


@contextmanager
def conn():
    with pool().connection() as c:
        yield c


def q(sql: str, params=None) -> list[dict]:
    with conn() as c:
        cur = c.execute(sql, params)
        return cur.fetchall() if cur.description else []


def q1(sql: str, params=None) -> dict | None:
    rows = q(sql, params)
    return rows[0] if rows else None


def ex(sql: str, params=None) -> int:
    with conn() as c:
        return c.execute(sql, params).rowcount


def init_schema() -> None:
    sql = (Path(__file__).parent / "schema.sql").read_text()
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as c:
        c.execute(sql)
