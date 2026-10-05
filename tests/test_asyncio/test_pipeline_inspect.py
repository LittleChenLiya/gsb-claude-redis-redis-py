"""Tests for offline async Pipeline.inspect()."""

from unittest import mock

import pytest
from redis.asyncio.client import Pipeline
from redis.commands.specs import InspectedCommand, SensitiveValue


@pytest.fixture
def mock_async_connection():
    """A fake async connection that never touches a real socket."""
    conn = mock.MagicMock()
    conn.host = "localhost"
    conn.port = 6379
    conn.db = 0

    async def mock_call_with_retry(
        do, fail, is_retryable=None, with_failure_count=False
    ):
        return await do()

    conn.retry = mock.MagicMock()
    conn.retry.call_with_retry = mock_call_with_retry
    conn.retry.get_retries.return_value = 0
    return conn


@pytest.fixture
def mock_async_connection_pool(mock_async_connection):
    """A fake async connection pool that hands out the fake connection."""
    pool = mock.MagicMock()
    pool.get_connection = mock.AsyncMock(return_value=mock_async_connection)
    pool.release = mock.AsyncMock()
    pool.get_encoder.return_value = mock.MagicMock()
    pool.get_protocol.return_value = 2
    return pool


def make_async_pipeline(mock_async_connection_pool, transaction=True):
    """Build an async Pipeline backed by the fake connection pool."""
    return Pipeline(
        connection_pool=mock_async_connection_pool,
        response_callbacks={},
        transaction=transaction,
        shard_hint=None,
    )


@pytest.mark.asyncio
class TestAsyncPipelineInspect:
    """Tests for async Pipeline.inspect() using a fake connection pool."""

    async def test_inspect_get_known_key(self, mock_async_connection_pool):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.get("mykey")
        report = pipe.inspect()
        assert len(report) == 1
        entry = report[0]
        assert entry.command == "GET"
        assert entry.key_positions == (1,)
        assert entry.index == 0

    async def test_inspect_set_single_key(self, mock_async_connection_pool):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.set("foo", "bar")
        report = pipe.inspect()
        assert len(report) == 1
        entry = report[0]
        assert entry.command == "SET"
        # SET has exactly one key at position 1; the value is not a key.
        assert entry.key_positions == (1,)

    async def test_inspect_mset_known_keys(self, mock_async_connection_pool):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.mset({"k1": "v1", "k2": "v2"})
        report = pipe.inspect()
        assert len(report) == 1
        entry = report[0]
        assert entry.command == "MSET"
        assert entry.key_positions == (1, 3)

    async def test_inspect_custom_command_unknown(self, mock_async_connection_pool):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.execute_command("MYCUSTOMCMD", "a", "b")
        report = pipe.inspect()
        assert len(report) == 1
        entry = report[0]
        assert entry.command == "MYCUSTOMCMD"
        assert entry.key_positions is None

    async def test_inspect_password_not_leaked(self, mock_async_connection_pool):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.execute_command("AUTH", "supersecret")
        report = pipe.inspect()
        entry = report[0]
        assert "supersecret" not in repr(report)
        assert isinstance(entry.args[1], SensitiveValue)

    async def test_inspect_does_not_change_command_stack(
        self,
        mock_async_connection_pool,
    ):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.get("a").set("b", "c")
        before = list(pipe.command_stack)
        pipe.inspect()
        after = list(pipe.command_stack)
        assert before == after
        assert len(pipe.command_stack) == 2

    async def test_inspect_does_not_send_commands(
        self,
        mock_async_connection_pool,
        mock_async_connection,
    ):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.get("a")
        pipe.inspect()
        mock_async_connection.send_command.assert_not_called()
        mock_async_connection.send_packed_command.assert_not_called()

    async def test_inspect_transaction_order_with_multi_exec(
        self,
        mock_async_connection_pool,
    ):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=True)
        pipe.get("a").set("b", "c")
        report = pipe.inspect()
        # MULTI, GET, SET, EXEC
        assert len(report) == 4
        assert report[0].command == "MULTI"
        assert report[0].is_boundary is True
        assert report[0].index is None
        assert report[0].in_transaction is True
        assert report[1].command == "GET"
        assert report[1].in_transaction is True
        assert report[2].command == "SET"
        assert report[2].in_transaction is True
        assert report[3].command == "EXEC"
        assert report[3].is_boundary is True
        assert report[3].index is None
        assert report[3].in_transaction is True

    async def test_inspect_no_transaction_when_disabled(
        self,
        mock_async_connection_pool,
    ):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.get("a")
        report = pipe.inspect()
        assert len(report) == 1
        assert report[0].command == "GET"
        assert report[0].in_transaction is False

    async def test_inspect_explicit_transaction(self, mock_async_connection_pool):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.multi()
        pipe.get("a")
        report = pipe.inspect()
        assert len(report) == 3
        assert report[0].command == "MULTI"
        assert report[1].command == "GET"
        assert report[1].in_transaction is True
        assert report[2].command == "EXEC"

    async def test_inspect_preserves_original_indices(
        self,
        mock_async_connection_pool,
    ):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.get("a").set("b", "c").mset({"d": "e", "f": "g"})
        report = pipe.inspect()
        assert [e.index for e in report] == [0, 1, 2]
        assert [e.command for e in report] == ["GET", "SET", "MSET"]

    async def test_inspect_empty_pipeline(self, mock_async_connection_pool):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        report = pipe.inspect()
        assert report == []

    async def test_inspect_empty_transaction_pipeline(
        self,
        mock_async_connection_pool,
    ):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=True)
        report = pipe.inspect()
        # Even an empty transaction reports the boundaries.
        assert len(report) == 2
        assert report[0].command == "MULTI"
        assert report[1].command == "EXEC"

    async def test_inspect_output_protocol_is_list_of_inspected_commands(
        self,
        mock_async_connection_pool,
    ):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.get("a")
        report = pipe.inspect()
        assert isinstance(report, list)
        assert all(isinstance(e, InspectedCommand) for e in report)

    async def test_inspect_does_not_trigger_encode(self, mock_async_connection_pool):
        pipe = make_async_pipeline(mock_async_connection_pool, transaction=False)
        pipe.get("a")
        # Encoder is only touched when commands are actually sent.
        pipe.inspect()
        mock_async_connection_pool.get_encoder.assert_not_called()
