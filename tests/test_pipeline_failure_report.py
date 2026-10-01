"""
Tests for pipeline partial-failure localization (``report_failures``).

Every scenario uses a fully scripted in-process connection -- no Redis
server is required. The fake connection replays a predefined response
stream, so the third command can be made to fail or the socket can be made
to break at an exact point.

Scenarios covered
-----------------
* normal path (all commands succeed),
* third-command error in both non-transactional and transactional pipelines,
* connection break mid-pipeline (transport-level failure),
* legacy mode (default behavior: annotated exception raised, transaction
  rollback / WatchError semantics unchanged),
* secret-safety of the failure report.
"""

import pytest

import redis
from redis.backoff import NoBackoff
from redis.client import PipelineFailure, PipelineResult
from redis.connection import Connection, ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.retry import Retry


class ScriptedConnection(Connection):
    """
    A connection that replays a scripted response stream instead of talking
    to a server.

    Each ``read_response`` call yields the next scripted item: either a raw
    reply (``bytes`` / Python value) which is returned, or an exception
    instance which is raised. ``break_after`` simulates a dead socket: after
    that many successful replies every further read raises a
    :class:`~redis.exceptions.ConnectionError`.
    """

    def __init__(self, *args, script=(), break_after=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._script = list(script)
        self._cursor = 0
        self.break_after = break_after
        self.sent_commands = []
        self.disconnect_count = 0

    def connect(self):
        # No network: the scripted stream is already "connected".
        pass

    def can_read(self, timeout=0):
        return False

    def pack_commands(self, commands):
        # commands is a list of arg tuples (the callers strip options)
        self.sent_commands.append([tuple(args) for args in commands])
        return b"packed"

    def send_packed_command(self, command, check_health=True):
        pass

    def read_response(self, disable_decoding=False, **kwargs):
        if self.break_after is not None and self._cursor >= self.break_after:
            raise RedisConnectionError("fake connection broken")
        item = self._script[self._cursor]
        self._cursor += 1
        if isinstance(item, BaseException):
            raise item
        return item

    def disconnect(self, *args, **kwargs):
        self.disconnect_count += 1
        self._cursor = 0


def make_client(script=(), break_after=None, transaction=True):
    pool = ConnectionPool(
        connection_class=ScriptedConnection,
        script=script,
        break_after=break_after,
        retry=Retry(NoBackoff(), 0),
    )
    return redis.Redis(connection_pool=pool), pool


class TestPipelineFailureReport:
    def test_all_succeed_report_mode(self):
        r, pool = make_client(script=[b"OK", b"1", 1])
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", "1").get("a").lpush("x", 1)
            result = pipe.execute(report_failures=True)

        assert isinstance(result, PipelineResult)
        assert result.succeeded
        assert result.failed_indices == []
        assert result.failures == []
        assert result.responses == [True, b"1", 1]
        # the pipe was reset and the connection returned to the pool
        assert len(pipe) == 0
        assert pipe.connection is None
        assert len(pool._in_use_connections) == 0

    def test_third_command_fails_nontransactional(self):
        error = ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )
        r, pool = make_client(script=[b"OK", b"OK", error, b"OK"])
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", 1).set("b", 2).lpush("c", 3).set("d", 4)
            result = pipe.execute(report_failures=True)

        assert not result.succeeded
        assert result.failed_indices == [2]
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert isinstance(failure, PipelineFailure)
        assert failure.index == 2
        assert failure.command == "LPUSH"
        assert failure.error is error
        assert "WRONGTYPE" in failure.message
        # the failing entry in responses is the raw server error, not an
        # annotated exception
        assert result.responses[2] is error
        # successful entries are still parsed
        assert result.responses[0] is True
        assert result.responses[1] is True
        assert result.responses[3] == 1
        # the pipe was reset and the connection returned to the pool
        assert len(pipe) == 0
        assert pipe.connection is None
        assert len(pool._in_use_connections) == 0

    def test_third_command_fails_transactional(self):
        error = ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )
        # MULTI -> OK, QUEUED x4, EXEC -> [OK, OK, error, OK]
        r, pool = make_client(
            script=[
                b"OK",
                b"QUEUED",
                b"QUEUED",
                b"QUEUED",
                b"QUEUED",
                [b"OK", b"OK", error, b"OK"],
            ]
        )
        with r.pipeline(transaction=True) as pipe:
            pipe.set("a", 1).set("b", 2).lpush("c", 3).set("d", 4)
            result = pipe.execute(report_failures=True)

        assert result.failed_indices == [2]
        assert result.failures[0].command == "LPUSH"
        assert result.responses[2] is error
        assert result.responses[0] is True
        assert result.responses[1] is True
        assert result.responses[3] is True
        # the transaction was sent as a single atomic MULTI/EXEC batch
        conn = pool._available_connections[0]
        assert conn.sent_commands[0][0] == ("MULTI",)
        assert conn.sent_commands[0][-1] == ("EXEC",)
        assert conn.sent_commands[0] == [
            ("MULTI",),
            ("SET", "a", 1),
            ("SET", "b", 2),
            ("LPUSH", "c", 3),
            ("SET", "d", 4),
            ("EXEC",),
        ]
        # the pipe was reset and the connection returned to the pool
        assert len(pipe) == 0
        assert pipe.connection is None
        assert len(pool._in_use_connections) == 0

    def test_report_mode_ignores_raise_on_error(self):
        # report_failures=True reports partial failures even when the
        # default raise_on_error=True is left untouched
        error = ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )
        r, pool = make_client(script=[b"OK", error])
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", 1).lpush("c", 3)
            result = pipe.execute(raise_on_error=True, report_failures=True)

        assert isinstance(result, PipelineResult)
        assert result.failed_indices == [1]
        assert result.failures[0].command == "LPUSH"

    def test_report_does_not_leak_command_secrets(self):
        secret = "s3cr3t-requirepass-value"
        error = ResponseError("ERR unknown command 'CONFIG'")
        r, pool = make_client(script=[b"OK", b"OK", error, b"OK"])
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", 1).set("b", 2).config_set("requirepass", secret).set("d", 4)
            result = pipe.execute(report_failures=True)

        assert result.failed_indices == [2]
        failure = result.failures[0]
        assert failure.command == "CONFIG SET"
        # the report carries the command name but never its arguments
        assert secret not in repr(failure)
        assert secret not in str(failure)
        assert secret not in repr(result)
        assert secret not in str(result)
        # the exception in responses is the raw server error, not an
        # annotated one that echoes the full command
        assert result.responses[2] is error
        assert secret not in str(result.responses[2])

    def test_legacy_mode_annotates_full_command(self):
        # Default behavior (no report_failures) is unchanged: the raised
        # exception message carries the full command text, as before.
        secret = "s3cr3t-requirepass-value"
        r, pool = make_client(
            script=[
                b"OK",
                b"OK",
                ResponseError("ERR unknown command 'CONFIG'"),
                b"OK",
            ]
        )
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", 1).set("b", 2).config_set("requirepass", secret).set("d", 4)
            with pytest.raises(ResponseError) as exc_info:
                pipe.execute()
            message = str(exc_info.value)
            assert message.startswith(
                "Command # 3 (CONFIG SET requirepass "
                f"{secret}) of pipeline caused error: "
            )
        # the pipe was reset and the connection returned to the pool
        assert len(pipe) == 0
        assert pipe.connection is None

    def test_legacy_mode_returns_exceptions_inline(self):
        # raise_on_error=False (legacy) inlines the exception into the
        # response list instead of raising
        error = ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )
        r, pool = make_client(script=[b"OK", b"OK", error, b"OK"])
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", 1).set("b", 2).lpush("c", 3).set("d", 4)
            result = pipe.execute(raise_on_error=False)

        assert isinstance(result, list)
        assert result[0] is True
        assert result[1] is True
        assert result[2] is error
        assert result[3] == 1

    def test_connection_break_nontransactional(self):
        r, pool = make_client(script=[b"OK", b"OK", b"OK", b"OK"], break_after=2)
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", 1).set("b", 2).set("c", 3).set("d", 4)
            with pytest.raises(RedisConnectionError, match="fake connection broken"):
                pipe.execute(report_failures=True)

        # no partial report is fabricated for a transport-level failure;
        # the pipe is reset and the connection returned to the pool
        assert len(pipe) == 0
        assert pipe.connection is None
        assert len(pool._in_use_connections) == 0
        conn = pool._available_connections[0]
        assert conn.disconnect_count >= 1

    def test_connection_break_transactional(self):
        r, pool = make_client(
            script=[
                b"OK",
                b"QUEUED",
                b"QUEUED",
                b"QUEUED",
                b"QUEUED",
                [b"OK", b"OK", b"OK", b"OK"],
            ],
            break_after=2,
        )
        with r.pipeline(transaction=True) as pipe:
            pipe.set("a", 1).set("b", 2).set("c", 3).set("d", 4)
            with pytest.raises(RedisConnectionError, match="fake connection broken"):
                pipe.execute(report_failures=True)

        assert len(pipe) == 0
        assert pipe.connection is None
        assert len(pool._in_use_connections) == 0

    def test_watched_key_change_still_raises_watch_error(self):
        # EXEC returns nil because the watched key changed; the WatchError
        # rollback path is unchanged by report_failures
        r, pool = make_client(script=[b"OK", b"OK", b"QUEUED", None])
        with r.pipeline(transaction=True) as pipe:
            pipe.watch("a")
            pipe.set("a", 1)
            with pytest.raises(redis.WatchError, match="Watched variable changed"):
                pipe.execute(report_failures=True)

        assert len(pipe) == 0
        assert pipe.connection is None
        assert len(pool._in_use_connections) == 0

    def test_pipeline_result_sequence_protocol(self):
        error = ResponseError("boom")
        r, pool = make_client(script=[b"OK", error])
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", 1).lpush("c", 3)
            result = pipe.execute(report_failures=True)

        assert len(result) == 2
        assert result[0] is True
        assert result[1] is error
        assert list(result) == [True, error]
        assert result.failed(0) is False
        assert result.failed(1) is True
        assert result.failed_indices == [1]

    def test_pipe_reusable_after_report(self):
        # the pipe is restored to a working state after a reported failure
        error = ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )
        r, pool = make_client(script=[b"OK", error, b"OK"])
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", 1).lpush("c", 3)
            result = pipe.execute(report_failures=True)
            assert result.failed_indices == [1]
            # a subsequent execute on the same pipe works
            pipe.set("z", "zzz")
            assert pipe.execute() == [True]
