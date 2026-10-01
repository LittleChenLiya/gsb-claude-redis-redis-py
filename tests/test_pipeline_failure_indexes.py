"""Acceptance tests for the optional pipeline failure-index report.

These tests exercise ``redis.client.Pipeline`` against a *local, controllable*
fixture: a fake connection backed by a scripted RESP response stream (no real
Redis server, no Docker). They cover both the happy path and the failure
paths:

* a normal run reports ``[]`` and the plain-list return type keeps working
  (backwards compatibility);
* a ``ResponseError`` on the third queued command is reported at index 2,
  both for the transactional (MULTI/EXEC, rollback semantics) and the
  plain-pipeline path;
* a server that drops the connection mid-stream is reported via
  ``pipeline_command_index`` on the raised connection error;
* the raised error never carries command arguments (secrets).
"""

from unittest import mock

import pytest

from redis._parsers.resp2 import _RESP2Parser
from redis.backoff import NoBackoff
from redis.client import Pipeline
from redis.connection import Connection
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.retry import Retry

# The SET values are sentinels proving the error metadata never leaks
# command arguments.
SECRET = "super-secret-token-value"
QUEUED_COMMANDS = (
    ("SET", "k1", SECRET),
    ("GET", "k1"),
    ("SET", "k2", SECRET),
    ("GET", "k2"),
)


class FakeSocket:
    """A readable socket fed from a scripted RESP byte stream."""

    def __init__(self, data=b""):
        self.data = bytearray(data)
        self.pos = 0
        self.sent = bytearray()

    def recv(self, bufsize):
        result = bytes(self.data[self.pos : self.pos + bufsize])
        self.pos += len(result)
        return result

    def recv_into(self, buffer, nbytes=0, flags=0):
        if nbytes == 0:
            nbytes = len(buffer)
        result = bytes(self.data[self.pos : self.pos + nbytes])
        self.pos += len(result)
        buffer[: len(result)] = result
        return len(result)

    def sendall(self, data):
        self.sent += data

    def shutdown(self, flags):
        pass

    def close(self):
        pass

    def getsockname(self):
        return ("127.0.0.1", 0)


def make_pipeline(response_stream, transaction=True):
    """
    Build a Pipeline whose single connection is a real ``Connection`` using
    the pure-Python RESP2 parser, with its socket replaced by a scripted
    ``FakeSocket``. Writes are swallowed; reads come from ``response_stream``.
    """
    conn = Connection(protocol=2, parser_class=_RESP2Parser, socket_read_size=16)
    # No retries: a dropped stream must surface immediately and
    # deterministically.
    conn.retry = Retry(NoBackoff(), 0)
    conn.host = "localhost"
    conn.port = 6379
    conn.db = 0

    fake_sock = FakeSocket(response_stream)
    conn._sock = fake_sock
    conn._parser.on_connect(conn)  # bind the SocketBuffer to the fake socket

    pool = mock.MagicMock()
    pool.get_connection.return_value = conn
    pool.release.return_value = None

    pipe = Pipeline(
        connection_pool=pool,
        response_callbacks={},
        transaction=transaction,
        shard_hint=None,
    )
    return pipe, conn, fake_sock


def queue_commands(pipe):
    for command_args in QUEUED_COMMANDS:
        pipe.execute_command(*command_args)


def queued_replies(count=None):
    """``+QUEUED`` replies read between MULTI and EXEC."""
    return b"+QUEUED\r\n" * (len(QUEUED_COMMANDS) if count is None else count)


def assert_no_secret(exception):
    """
    Structured failure metadata must never expose command arguments.

    The legacy annotation *message* keeps a truncated rendering of the
    command for backwards compatibility; the new machine-readable report
    fields (index / command name) must contain no arguments.
    """
    assert SECRET not in getattr(exception, "pipeline_command_name", "")
    assert isinstance(exception.pipeline_command_index, int)


class TestPipelineFailureIndexesTransaction:
    def test_success_returns_no_indexes_and_wires_multi_exec(self):
        # MULTI -> +OK; 4 x +QUEUED; EXEC -> [1, v1, 1, v2]
        stream = (
            b"+OK\r\n"
            + queued_replies()
            + b"*4\r\n:1\r\n$2\r\nv1\r\n:1\r\n$2\r\nv2\r\n"
        )
        pipe, conn, sock = make_pipeline(stream, transaction=True)
        queue_commands(pipe)

        responses, failed = pipe.execute(
            raise_on_error=False, return_failure_indexes=True
        )

        # key results
        assert responses == [1, b"v1", 1, b"v2"]
        assert failed == []
        # the commands actually executed: MULTI ... EXEC frame was written
        sent = bytes(sock.sent)
        assert sent.startswith(b"*1\r\n$5\r\nMULTI\r\n")
        assert sent.rstrip().endswith(b"*1\r\n$4\r\nEXEC")
        assert b"SET" in sent and b"GET" in sent

    def test_third_command_error_is_index_two_and_rolls_back(self):
        # Redis embeds per-command errors inside the EXEC array. Commands 1,
        # 2 and 4 reply OK; command 3 (SET k2) failed server-side, so the
        # server applies none of the writes (MULTI/EXEC rollback semantics).
        stream = (
            b"+OK\r\n"
            + queued_replies()
            + b"*4\r\n:1\r\n$2\r\nv1\r\n-ERR WRONGTYPE\r\n:1\r\n"
        )
        pipe, conn, sock = make_pipeline(stream, transaction=True)
        queue_commands(pipe)

        responses, failed = pipe.execute(
            raise_on_error=False, return_failure_indexes=True
        )

        assert responses[0] == 1
        assert responses[1] == b"v1"
        assert isinstance(responses[2], ResponseError)
        assert responses[3] == 1
        assert failed == [2]
        # the embedded exception is annotated without arguments
        err = responses[2]
        assert err.pipeline_command_index == 2
        assert err.pipeline_command_name == "SET"
        assert_no_secret(err)
        # transaction framing is untouched -> server-side rollback semantics
        sent = bytes(sock.sent)
        assert b"MULTI" in sent and b"EXEC" in sent

    def test_default_mode_raises_annotated_error_for_third_command(self):
        # Default raise_on_error=True: raises the index-2 exception, keeps
        # the historical message prefix, and exposes the index attribute.
        stream = (
            b"+OK\r\n"
            + queued_replies()
            + b"*4\r\n:1\r\n$2\r\nv1\r\n-ERR WRONGTYPE\r\n:1\r\n"
        )
        pipe, conn, sock = make_pipeline(stream, transaction=True)
        queue_commands(pipe)

        with pytest.raises(ResponseError) as exc_info:
            result = pipe.execute()  # legacy behavior: plain list returned
            # on success this would be a list; on error the line above raises
            assert isinstance(result, list)

        assert str(exc_info.value).startswith("Command # 3 (SET")
        assert exc_info.value.pipeline_command_index == 2
        assert exc_info.value.pipeline_command_name == "SET"
        assert_no_secret(exc_info.value)

    def test_exec_abort_tags_rejected_queued_command(self):
        # A command rejected *at queue time* makes the server fail EXEC:
        # Redis aborts and discards the whole transaction. The raised error
        # must identify the offending command (command #3 -> index 2).
        stream = (
            b"+OK\r\n"
            + queued_replies(2)
            + b"-ERR unknown command 'BOGUS'\r\n"
            + queued_replies(1)
            + b"-EXECABORT Transaction discarded because of previous errors.\r\n"
        )
        pipe, conn, sock = make_pipeline(stream, transaction=True)
        pipe.execute_command("SET", "k1", SECRET)
        pipe.execute_command("GET", "k1")
        pipe.execute_command("BOGUS", SECRET)
        pipe.execute_command("GET", "k2")

        with pytest.raises(ResponseError) as exc_info:
            pipe.execute()

        assert exc_info.value.pipeline_command_index == 2
        assert exc_info.value.pipeline_command_name == "BOGUS"
        assert_no_secret(exc_info.value)


class TestPipelineFailureIndexesNoTransaction:
    def test_success_plain_pipeline(self):
        stream = b":1\r\n$2\r\nv1\r\n:1\r\n$2\r\nv2\r\n"
        pipe, conn, sock = make_pipeline(stream, transaction=False)
        queue_commands(pipe)

        responses, failed = pipe.execute(
            raise_on_error=False, return_failure_indexes=True
        )
        assert responses == [1, b"v1", 1, b"v2"]
        assert failed == []
        # no transaction frame in a plain pipeline
        sent = bytes(sock.sent)
        assert b"MULTI" not in sent and b"EXEC" not in sent

    def test_third_command_error_is_index_two(self):
        # replies arrive one per command; the third reply is an error.
        stream = b":1\r\n$2\r\nv1\r\n-ERR WRONGTYPE\r\n:1\r\n"
        pipe, conn, sock = make_pipeline(stream, transaction=False)
        queue_commands(pipe)

        responses, failed = pipe.execute(
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

    def test_connection_drop_during_third_reply(self):
        # Two replies are present, then the stream ends (server closed the
        # connection) while reading the third command's reply.
        stream = b":1\r\n$2\r\nv1\r\n"
        pipe, conn, sock = make_pipeline(stream, transaction=False)
        queue_commands(pipe)

        with pytest.raises(RedisConnectionError) as exc_info:
            pipe.execute()

        assert exc_info.value.pipeline_command_index == 2
        assert exc_info.value.pipeline_command_name == "SET"
        assert_no_secret(exc_info.value)
        # execute()'s finally still resets the pipeline
        assert len(pipe.command_stack) == 0

    def test_default_mode_without_report_is_unchanged(self):
        # raise_on_error=False, report off -> plain list (legacy shape)
        stream = b":1\r\n$2\r\nv1\r\n-ERR WRONGTYPE\r\n:1\r\n"
        pipe, conn, sock = make_pipeline(stream, transaction=False)
        queue_commands(pipe)

        result = pipe.execute(raise_on_error=False)
        assert isinstance(result, list)
        assert isinstance(result[2], ResponseError)

    def test_report_requires_raise_on_error_false(self):
        pipe, conn, sock = make_pipeline(b"", transaction=False)
        queue_commands(pipe)
        with pytest.raises(ValueError):
            pipe.execute(raise_on_error=True, return_failure_indexes=True)

    def test_empty_pipeline_report_shape(self):
        pipe, conn, sock = make_pipeline(b"", transaction=True)
        assert pipe.execute(return_failure_indexes=True) == ([], [])
        assert pipe.execute() == []


class TestPipelineFailureIndexesSecretSafety:
    @pytest.mark.parametrize("transaction", [True, False])
    def test_no_arguments_in_failure_report(self, transaction):
        if transaction:
            stream = (
                b"+OK\r\n"
                + queued_replies()
                + b"*4\r\n:1\r\n$2\r\nv1\r\n-ERR boom\r\n:1\r\n"
            )
        else:
            stream = b":1\r\n$2\r\nv1\r\n-ERR boom\r\n:1\r\n"
        pipe, conn, sock = make_pipeline(stream, transaction=transaction)
        queue_commands(pipe)
        responses, failed = pipe.execute(
            raise_on_error=False, return_failure_indexes=True
        )
        # the report itself contains plain integers only
        assert failed == [2]
        assert all(isinstance(i, int) for i in failed)
        assert SECRET not in repr(failed)
        assert SECRET not in responses[2].pipeline_command_name
        assert SECRET not in repr(failed)
