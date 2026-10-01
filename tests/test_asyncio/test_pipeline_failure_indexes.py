"""Acceptance tests for the optional async pipeline failure-index report.

Async mirror of ``tests/test_pipeline_failure_indexes.py``: exercises
``redis.asyncio.client.Pipeline`` against a local, scripted RESP response
stream (an ``asyncio.StreamReader`` stand-in, no real Redis server and no
Docker). Covers normal and failure paths for both the transactional
(MULTI/EXEC, rollback) and plain-pipeline modes, plus secret safety of the
structured report.
"""

import asyncio
from unittest import mock

import pytest

from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.asyncio.connection import Connection as AsyncConnection
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

# The SET values are sentinels proving the structured report never leaks
# command arguments.
SECRET = "super-secret-token-value"
QUEUED_COMMANDS = (
    ("SET", "k1", SECRET),
    ("GET", "k1"),
    ("SET", "k2", SECRET),
    ("GET", "k2"),
)


class _FakeTransport:
    def is_closing(self):
        return False


class FakeAsyncWriter:
    """Stand-in for asyncio.StreamWriter: records writes, never blocks."""

    def __init__(self):
        self.sent = bytearray()
        self.closed = False
        self.transport = _FakeTransport()

    def write(self, data):
        self.sent += data

    def writelines(self, data):
        for item in data:
            self.write(item)

    def drain(self):
        return asyncio.sleep(0)

    def close(self):
        self.closed = True

    async def wait_closed(self):
        self.closed = True

    def is_closing(self):
        return self.closed

    def can_write_eof(self):
        return False


def make_reader(data=b"", drop=False):
    """A StreamReader pre-loaded with ``data``; EOF simulates a dropped stream."""
    reader = asyncio.StreamReader()
    if data:
        reader.feed_data(data)
    # feed_eof whether or not data was supplied: after the scripted bytes are
    # consumed the connection reads EOF (server closed the connection).
    reader.feed_eof()
    return reader


def queued_replies(count=None):
    return b"+QUEUED\r\n" * (len(QUEUED_COMMANDS) if count is None else count)


def make_pipeline(response_stream, transaction=True):
    """
    Build an async Pipeline bound to a real async ``Connection`` whose reader
    is a scripted StreamReader and whose writer records (but never sends) the
    packed commands.
    """
    conn = AsyncConnection(protocol=2)
    conn.retry = AsyncRetry(NoBackoff(), 0)
    conn.host = "localhost"
    conn.port = 6379
    conn.db = 0

    writer = FakeAsyncWriter()
    conn._reader = make_reader(response_stream)
    conn._writer = writer
    conn._parser.on_connect(conn)

    pool = mock.MagicMock()

    async def get_connection(*args, **kwargs):
        return conn

    async def release(*args, **kwargs):
        return None

    pool.get_connection = get_connection
    pool.release = release

    pipe = AsyncPipeline(
        connection_pool=pool,
        response_callbacks={},
        transaction=transaction,
        shard_hint=None,
    )
    return pipe, conn, writer


def queue_commands(pipe):
    for command_args in QUEUED_COMMANDS:
        pipe.execute_command(*command_args)


def assert_no_secret(exception):
    assert SECRET not in getattr(exception, "pipeline_command_name", "")
    assert isinstance(exception.pipeline_command_index, int)


class TestAsyncPipelineFailureIndexesTransaction:
    async def test_success_returns_no_indexes_and_wires_multi_exec(self):
        stream = (
            b"+OK\r\n"
            + queued_replies()
            + b"*4\r\n:1\r\n$2\r\nv1\r\n:1\r\n$2\r\nv2\r\n"
        )
        pipe, conn, writer = make_pipeline(stream, transaction=True)
        queue_commands(pipe)

        responses, failed = await pipe.execute(
            raise_on_error=False, return_failure_indexes=True
        )
        assert responses == [1, b"v1", 1, b"v2"]
        assert failed == []
        sent = bytes(writer.sent)
        assert sent.startswith(b"*1\r\n$5\r\nMULTI\r\n")
        assert sent.rstrip().endswith(b"*1\r\n$4\r\nEXEC")

    async def test_third_command_error_is_index_two_and_rolls_back(self):
        # Per-command error inside the EXEC array -> index 2; transaction
        # framing guarantees server-side rollback.
        stream = (
            b"+OK\r\n"
            + queued_replies()
            + b"*4\r\n:1\r\n$2\r\nv1\r\n-ERR WRONGTYPE\r\n:1\r\n"
        )
        pipe, conn, writer = make_pipeline(stream, transaction=True)
        queue_commands(pipe)

        responses, failed = await pipe.execute(
            raise_on_error=False, return_failure_indexes=True
        )
        assert responses[0] == 1
        assert responses[1] == b"v1"
        assert isinstance(responses[2], ResponseError)
        assert responses[3] == 1
        assert failed == [2]
        err = responses[2]
        assert err.pipeline_command_index == 2
        assert err.pipeline_command_name == "SET"
        assert_no_secret(err)
        sent = bytes(writer.sent)
        assert b"MULTI" in sent and b"EXEC" in sent

    async def test_default_mode_raises_annotated_error_for_third_command(self):
        stream = (
            b"+OK\r\n"
            + queued_replies()
            + b"*4\r\n:1\r\n$2\r\nv1\r\n-ERR WRONGTYPE\r\n:1\r\n"
        )
        pipe, conn, writer = make_pipeline(stream, transaction=True)
        queue_commands(pipe)

        with pytest.raises(ResponseError) as exc_info:
            await pipe.execute()

        assert str(exc_info.value).startswith("Command # 3 (SET")
        assert exc_info.value.pipeline_command_index == 2
        assert exc_info.value.pipeline_command_name == "SET"
        assert_no_secret(exc_info.value)

    async def test_exec_abort_tags_rejected_queued_command(self):
        stream = (
            b"+OK\r\n"
            + queued_replies(2)
            + b"-ERR unknown command 'BOGUS'\r\n"
            + queued_replies(1)
            + b"-EXECABORT Transaction discarded because of previous errors.\r\n"
        )
        pipe, conn, writer = make_pipeline(stream, transaction=True)
        pipe.execute_command("SET", "k1", SECRET)
        pipe.execute_command("GET", "k1")
        pipe.execute_command("BOGUS", SECRET)
        pipe.execute_command("GET", "k2")

        with pytest.raises(ResponseError) as exc_info:
            await pipe.execute()

        assert exc_info.value.pipeline_command_index == 2
        assert exc_info.value.pipeline_command_name == "BOGUS"
        assert_no_secret(exc_info.value)


class TestAsyncPipelineFailureIndexesNoTransaction:
    async def test_success_plain_pipeline(self):
        stream = b":1\r\n$2\r\nv1\r\n:1\r\n$2\r\nv2\r\n"
        pipe, conn, writer = make_pipeline(stream, transaction=False)
        queue_commands(pipe)

        responses, failed = await pipe.execute(
            raise_on_error=False, return_failure_indexes=True
        )
        assert responses == [1, b"v1", 1, b"v2"]
        assert failed == []
        sent = bytes(writer.sent)
        assert b"MULTI" not in sent and b"EXEC" not in sent

    async def test_third_command_error_is_index_two(self):
        stream = b":1\r\n$2\r\nv1\r\n-ERR WRONGTYPE\r\n:1\r\n"
        pipe, conn, writer = make_pipeline(stream, transaction=False)
        queue_commands(pipe)

        responses, failed = await pipe.execute(
            raise_on_error=False, return_failure_indexes=True
        )
        assert isinstance(responses[2], ResponseError)
        assert failed == [2]
        err = responses[2]
        assert err.pipeline_command_index == 2
        assert err.pipeline_command_name == "SET"
        assert_no_secret(err)

    async def test_connection_drop_during_third_reply(self):
        # Two replies scripted, then EOF while reading the third.
        stream = b":1\r\n$2\r\nv1\r\n"
        pipe, conn, writer = make_pipeline(stream, transaction=False)
        queue_commands(pipe)

        with pytest.raises(RedisConnectionError) as exc_info:
            await pipe.execute()

        assert exc_info.value.pipeline_command_index == 2
        assert exc_info.value.pipeline_command_name == "SET"
        assert_no_secret(exc_info.value)
        # execute()'s finally still resets the pipeline
        assert len(pipe.command_stack) == 0

    async def test_default_mode_without_report_is_unchanged(self):
        stream = b":1\r\n$2\r\nv1\r\n-ERR WRONGTYPE\r\n:1\r\n"
        pipe, conn, writer = make_pipeline(stream, transaction=False)
        queue_commands(pipe)

        result = await pipe.execute(raise_on_error=False)
        assert isinstance(result, list)
        assert isinstance(result[2], ResponseError)

    async def test_report_requires_raise_on_error_false(self):
        pipe, conn, writer = make_pipeline(b"", transaction=False)
        queue_commands(pipe)
        with pytest.raises(ValueError):
            await pipe.execute(raise_on_error=True, return_failure_indexes=True)

    async def test_empty_pipeline_report_shape(self):
        pipe, conn, writer = make_pipeline(b"", transaction=True)
        assert await pipe.execute(return_failure_indexes=True) == ([], [])
        assert await pipe.execute() == []
