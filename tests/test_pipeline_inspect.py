"""Tests for offline Pipeline.inspect()."""

from unittest import mock
import pytest
from redis.client import Pipeline
from redis.commands.specs import (
    InspectedCommand,
    SensitiveValue,
    get_command_spec,
    inspect_command,
)


@pytest.fixture
def mock_connection():
    """A fake connection that never touches a real socket."""
    conn = mock.MagicMock()
    conn.host = "localhost"
    conn.port = 6379
    conn.db = 0

    def mock_call_with_retry(do, fail, is_retryable=None, with_failure_count=False):
        return do()

    conn.retry.call_with_retry = mock_call_with_retry
    return conn


@pytest.fixture
def mock_connection_pool(mock_connection):
    """A fake connection pool that hands out the fake connection."""
    pool = mock.MagicMock()
    pool.get_connection.return_value = mock_connection
    pool.get_encoder.return_value = mock.MagicMock()
    return pool


def make_pipeline(mock_connection_pool, transaction=True):
    """Build a Pipeline backed by the fake connection pool."""
    return Pipeline(
        connection_pool=mock_connection_pool,
        response_callbacks={},
        transaction=transaction,
        shard_hint=None,
    )


class TestInspectCommandSpecs:
    """Unit tests for the registered command specs and the pure inspect_command helper."""

    def test_get_has_known_key_position(self):
        spec = get_command_spec("GET")
        assert spec is not None
        assert spec.first_key_pos == 1
        assert spec.key_step == 1

    def test_mset_has_stepped_key_positions(self):
        spec = get_command_spec("MSET")
        assert spec is not None
        assert spec.first_key_pos == 1
        assert spec.key_step == 2

    def test_unknown_command_returns_none(self):
        assert get_command_spec("MYCUSTOMCMD") is None

    def test_inspect_get_reports_known_key(self):
        entry = inspect_command(0, ("GET", "mykey"), in_transaction=False)
        assert entry.command == "GET"
        assert entry.key_positions == (1,)
        assert entry.args == ("GET", "mykey")
        assert entry.index == 0
        assert entry.in_transaction is False
        assert entry.is_boundary is False

    def test_inspect_mset_reports_stepped_keys(self):
        entry = inspect_command(
            1, ("MSET", "k1", "v1", "k2", "v2"), in_transaction=False
        )
        assert entry.key_positions == (1, 3)

    def test_inspect_unknown_command_reports_none(self):
        entry = inspect_command(2, ("MYCUSTOMCMD", "a", "b"), in_transaction=False)
        assert entry.key_positions is None
        assert entry.command == "MYCUSTOMCMD"

    def test_inspect_does_not_mutate_args(self):
        args = ("GET", "mykey")
        inspect_command(0, args, in_transaction=False)
        assert args == ("GET", "mykey")

    def test_auth_password_is_redacted(self):
        entry = inspect_command(0, ("AUTH", "supersecret"), in_transaction=False)
        assert len(entry.args) == 2
        assert entry.args[0] == "AUTH"
        assert isinstance(entry.args[1], SensitiveValue)
        assert entry.args[1].category == "password"
        assert "supersecret" not in repr(entry.args)
        assert "supersecret" not in str(entry.args)

    def test_config_set_requirepass_is_redacted(self):
        entry = inspect_command(
            0, ("CONFIG", "SET", "requirepass", "hunter2"), in_transaction=False
        )
        assert isinstance(entry.args[3], SensitiveValue)
        assert entry.args[3].category == "password"
        assert "hunter2" not in repr(entry.args)

    def test_config_set_other_value_is_not_redacted(self):
        entry = inspect_command(
            0, ("CONFIG", "SET", "timeout", "100"), in_transaction=False
        )
        assert entry.args[3] == "100"

    def test_inspect_command_is_frozen(self):
        entry = inspect_command(0, ("GET", "k"), in_transaction=False)
        with pytest.raises(Exception):
            entry.command = "SET"


class TestPipelineInspect:
    """Tests for Pipeline.inspect() using a fake connection pool."""

    def test_inspect_get_known_key(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.get("mykey")
        report = pipe.inspect()
        assert len(report) == 1
        entry = report[0]
        assert entry.command == "GET"
        assert entry.key_positions == (1,)
        assert entry.index == 0

    def test_inspect_set_single_key(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.set("foo", "bar")
        report = pipe.inspect()
        assert len(report) == 1
        entry = report[0]
        assert entry.command == "SET"
        # SET has exactly one key at position 1; the value is not a key.
        assert entry.key_positions == (1,)

    def test_inspect_mset_known_keys(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.mset({"k1": "v1", "k2": "v2"})
        report = pipe.inspect()
        assert len(report) == 1
        entry = report[0]
        assert entry.command == "MSET"
        assert entry.key_positions == (1, 3)

    def test_inspect_custom_command_unknown(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.execute_command("MYCUSTOMCMD", "a", "b")
        report = pipe.inspect()
        assert len(report) == 1
        entry = report[0]
        assert entry.command == "MYCUSTOMCMD"
        assert entry.key_positions is None

    def test_inspect_password_not_leaked(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.execute_command("AUTH", "supersecret")
        report = pipe.inspect()
        entry = report[0]
        assert "supersecret" not in repr(report)
        assert isinstance(entry.args[1], SensitiveValue)

    def test_inspect_does_not_change_command_stack(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.get("a").set("b", "c")
        before = list(pipe.command_stack)
        pipe.inspect()
        after = list(pipe.command_stack)
        assert before == after
        assert len(pipe.command_stack) == 2

    def test_inspect_does_not_send_commands(
        self, mock_connection_pool, mock_connection
    ):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.get("a")
        pipe.inspect()
        mock_connection.send_command.assert_not_called()
        mock_connection.send_packed_command.assert_not_called()

    def test_inspect_transaction_order_with_multi_exec(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=True)
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

    def test_inspect_no_transaction_when_disabled(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.get("a")
        report = pipe.inspect()
        assert len(report) == 1
        assert report[0].command == "GET"
        assert report[0].in_transaction is False

    def test_inspect_explicit_transaction(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.multi()
        pipe.get("a")
        report = pipe.inspect()
        assert len(report) == 3
        assert report[0].command == "MULTI"
        assert report[1].command == "GET"
        assert report[1].in_transaction is True
        assert report[2].command == "EXEC"

    def test_inspect_preserves_original_indices(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.get("a").set("b", "c").mset({"d": "e", "f": "g"})
        report = pipe.inspect()
        assert [e.index for e in report] == [0, 1, 2]
        assert [e.command for e in report] == ["GET", "SET", "MSET"]

    def test_inspect_empty_pipeline(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        report = pipe.inspect()
        assert report == []

    def test_inspect_empty_transaction_pipeline(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=True)
        report = pipe.inspect()
        # Even an empty transaction reports the boundaries.
        assert len(report) == 2
        assert report[0].command == "MULTI"
        assert report[1].command == "EXEC"

    def test_inspect_output_protocol_is_list_of_inspected_commands(
        self,
        mock_connection_pool,
    ):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.get("a")
        report = pipe.inspect()
        assert isinstance(report, list)
        assert all(isinstance(e, InspectedCommand) for e in report)

    def test_inspect_does_not_trigger_encode(self, mock_connection_pool):
        pipe = make_pipeline(mock_connection_pool, transaction=False)
        pipe.get("a")
        # Encoder is only touched when commands are actually sent.
        pipe.inspect()
        mock_connection_pool.get_encoder.assert_not_called()
