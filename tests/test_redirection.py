"""
Local, Docker-free tests for per-request cluster MOVED/ASK redirection tracing
(sync client counterpart of tests/test_asyncio/test_redirection.py).

The cluster is built against a fully mocked transport; deterministic MOVED/ASK
scripts are driven through it and both the final response and the redirection
report are asserted. No Redis server, Docker or network listener is involved.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from redis import Redis
from redis._parsers import CommandsParser
from redis.cluster import ClusterNode, RedisCluster
from redis.cluster import get_node_name
from redis.exceptions import AskError, ClusterError, MovedError
from redis.redirection import (
    RedirectionHop,
    RedirectionTracker,
    RedirectionType,
)

HOST = "127.0.0.1"

# 7000 owns every slot initially; 7001-7003 are cached members (listed as
# replicas) so every ASK/MOVED target resolves to a known node.
ALL_SLOTS_OWNED_BY_7000 = [
    [0, 4095, [HOST, 7000, "node_0"], [HOST, 7001, "node_1"]],
    [4096, 8191, [HOST, 7000, "node_0"], [HOST, 7002, "node_2"]],
    [8192, 12287, [HOST, 7000, "node_0"], [HOST, 7003, "node_3"]],
    [12288, 16383, [HOST, 7000, "node_0"], [HOST, 7001, "node_1"]],
]

SECRET_PASSWORD = "super-secret-password"


def _node(port: int) -> str:
    return get_node_name(host=HOST, port=port)


def _moved(slot: int, port: int) -> MovedError:
    return MovedError(f"{slot} {HOST}:{port}")


def _ask(slot: int, port: int) -> AskError:
    return AskError(f"{slot} {HOST}:{port}")


def _make_client(**kwargs: Any) -> RedisCluster:
    """Create a sync cluster client with a mocked transport and parser."""

    def init_execute_command(*args: Any, **options: Any) -> Any:
        command = args[0] if args else None
        if command == "CLUSTER SLOTS":
            return ALL_SLOTS_OWNED_BY_7000
        if command == "COMMAND":
            return {"get": [], "set": []}
        if command == "INFO":
            return {"cluster_enabled": True}
        if len(args) > 1 and args[1] == "cluster-require-full-coverage":
            return {"cluster-require-full-coverage": "yes"}
        raise AssertionError(f"unexpected command during initialization: {args}")

    with patch.object(Redis, "execute_command", side_effect=init_execute_command):
        with patch.object(
            CommandsParser, "initialize", autospec=True
        ) as cmd_parser_initialize:

            def cmd_init_mock(self: CommandsParser, r: ClusterNode) -> None:
                self.commands = {
                    "get": {
                        "name": "get",
                        "arity": 2,
                        "flags": ["readonly", "fast"],
                        "first_key_pos": 1,
                        "last_key_pos": 1,
                        "step_count": 1,
                    }
                }

            cmd_parser_initialize.side_effect = cmd_init_mock

            class MockedRedisCluster(RedisCluster):
                def __del__(self) -> None:
                    pass

            kwargs.setdefault("reinitialize_steps", 0)
            client = MockedRedisCluster(host=HOST, port=7000, **kwargs)

    # Give every node a no-op mock connection so the request path never opens
    # a socket; Redis.parse_response is what the scripts replace.
    for node in client.get_nodes():
        connection = Mock()
        connection.host = node.host
        connection.port = node.port
        connection.is_connected = True
        node.redis_connection.connection = connection

    return client


def _patch_responses(script, on_asking=None):
    """
    Patch Redis.parse_response so node identity comes from the
    connection the client picked. ``script(connection, command_name)`` returns
    the parsed response or raises a redirection; ASKING is answered with "OK"
    (and reported via the optional ``on_asking(port)`` callback).
    send_command is left untouched: each node carries a mock connection, so
    the write is a no-op while the read is driven entirely by the script.
    """
    parse_response = Mock()

    def parse_side_effect(connection, command_name, **options):
        if command_name == "ASKING":
            if on_asking is not None:
                on_asking(connection.port)
            return "OK"
        return script(connection, command_name)

    parse_response.side_effect = parse_side_effect
    return patch.object(Redis, "parse_response", parse_response)


@pytest.mark.fixed_client
def test_two_moved_one_ask_then_response():
    """
    Script for a GET on slot 12182:
      7000 -> MOVED 12182 7001
      7001 -> MOVED 12182 7002
      7002 -> ASK   12182 7003  (ASKING, then GET)
      7003 -> b"moved-asked-value"
    """
    client = _make_client(password=SECRET_PASSWORD)
    slot = 12182
    asking_nodes = []

    def script(connection, command_name):
        assert command_name == "GET"
        if connection.port == 7000:
            raise _moved(slot, 7001)
        if connection.port == 7001:
            raise _moved(slot, 7002)
        if connection.port == 7002:
            raise _ask(slot, 7003)
        assert connection.port == 7003
        return b"moved-asked-value"

    tracker = RedirectionTracker()
    with _patch_responses(script, on_asking=asking_nodes.append):
        result = client.execute_command("GET", "foo", redirect_tracker=tracker)

    client.close()

    assert result == b"moved-asked-value"
    assert asking_nodes == [7003]

    report = tracker.report
    assert report.final_node == _node(7003)
    assert report.moved_count == 2
    assert report.ask_count == 1
    assert report.total_redirections == 3
    assert report.redirect_loop_detected is False
    assert [hop.type for hop in report.hops] == [
        RedirectionType.MOVED,
        RedirectionType.MOVED,
        RedirectionType.ASK,
    ]
    assert report.hops[0] == RedirectionHop(
        type=RedirectionType.MOVED,
        slot_id=slot,
        from_node=_node(7000),
        to_node=_node(7001),
    )
    assert report.hops[1] == RedirectionHop(
        type=RedirectionType.MOVED,
        slot_id=slot,
        from_node=_node(7001),
        to_node=_node(7002),
    )
    assert report.hops[2] == RedirectionHop(
        type=RedirectionType.ASK,
        slot_id=slot,
        from_node=_node(7002),
        to_node=_node(7003),
    )
    # MOVED patched the slots cache; the ASK hop must not have.
    assert client.nodes_manager.get_node_from_slot(slot).name == _node(7002)

    print("\n--- two MOVED + one ASK report (sync) ---")
    print(report.summary())


@pytest.mark.fixed_client
def test_circular_moved_is_flagged_and_raises():
    """MOVED ping-pongs 7000 <-> 7001 until TTL exhaustion; loop is flagged."""
    client = _make_client()
    client.RedisClusterRequestTTL = 4
    slot = 12182

    def script(connection, command_name):
        assert command_name == "GET"
        target = 7001 if connection.port == 7000 else 7000
        raise _moved(slot, target)

    tracker = RedirectionTracker()
    with _patch_responses(script):
        with pytest.raises(ClusterError, match="TTL exhausted"):
            client.execute_command("GET", "foo", redirect_tracker=tracker)

    client.close()

    report = tracker.report
    assert report.final_node is None
    assert report.ask_count == 0
    assert report.moved_count == 4
    assert all(hop.type is RedirectionType.MOVED for hop in report.hops)
    assert report.redirect_loop_detected is True
    assert [hop.redirect_loop for hop in report.hops] == [
        False,
        False,
        True,
        True,
    ]
    assert [(hop.from_node, hop.to_node) for hop in report.hops] == [
        (_node(7000), _node(7001)),
        (_node(7001), _node(7000)),
        (_node(7000), _node(7001)),
        (_node(7001), _node(7000)),
    ]

    print("\n--- circular MOVED report (sync) ---")
    print(report.summary())


@pytest.mark.fixed_client
def test_ask_does_not_count_as_moved_or_loop():
    """An ASK redirection stays ASK and never flags a loop."""
    client = _make_client()
    slot = 12182

    def script(connection, command_name):
        assert command_name == "GET"
        if connection.port == 7000:
            raise _ask(slot, 7001)
        return b"ask-value"

    tracker = RedirectionTracker()
    with _patch_responses(script):
        result = client.execute_command("GET", "foo", redirect_tracker=tracker)

    client.close()

    assert result == b"ask-value"
    report = tracker.report
    assert report.moved_count == 0
    assert report.ask_count == 1
    assert report.redirect_loop_detected is False
    assert report.final_node == _node(7001)


@pytest.mark.fixed_client
def test_report_contains_no_authentication_information():
    """The tracker only records routing data; secrets must not leak into it."""
    client = _make_client(password=SECRET_PASSWORD, username="admin")
    slot = client.keyslot(SECRET_PASSWORD)

    def script(connection, command_name):
        if connection.port == 7000:
            raise _moved(slot, 7001)
        if connection.port == 7001:
            raise _ask(slot, 7002)
        return b"safe-value"

    tracker = RedirectionTracker()
    with _patch_responses(script):
        client.execute_command("GET", SECRET_PASSWORD, redirect_tracker=tracker)

    client.close()

    rendered = tracker.report.summary()
    assert SECRET_PASSWORD not in rendered
    assert "admin" not in rendered
    assert "password" not in rendered.lower()


@pytest.mark.fixed_client
def test_invalid_tracker_type_rejected():
    """A non-tracker object must be rejected instead of sent over the wire."""
    client = _make_client()
    with pytest.raises(TypeError, match="RedirectionTracker"):
        client.execute_command("GET", "foo", redirect_tracker=object())
    client.close()
