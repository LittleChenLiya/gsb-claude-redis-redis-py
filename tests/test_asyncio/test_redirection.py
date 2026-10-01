"""
Local, Docker-free tests for per-request cluster MOVED/ASK redirection tracing.

These tests build a RedisCluster against a fully mocked transport and drive
deterministic redirect scripts through it, then assert both the final response
and the redirection report. No Redis server, Docker or network listener is
involved.
"""

from typing import Any
from unittest import mock

import pytest

from redis._parsers import AsyncCommandsParser
from redis.asyncio.cluster import ClusterNode, RedisCluster
from redis.cluster import get_node_name
from redis.exceptions import AskError, ClusterError, MovedError
from redis.redirection import (
    RedirectionHop,
    RedirectionTracker,
    RedirectionType,
)

HOST = "127.0.0.1"

# Four known nodes. 7000 owns every slot initially (with 7001-7003 listed as
# replicas of the ranges) so ASK/MOVED targets are all cached cluster members.
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


async def _make_client(**kwargs: Any) -> RedisCluster:
    """Create a cluster client with a mocked transport and command parser."""
    init_slots = kwargs.pop("cluster_slots", ALL_SLOTS_OWNED_BY_7000)
    with mock.patch.object(ClusterNode, "execute_command") as execute_command_mock:

        async def execute_command(*args: Any, **options: Any) -> Any:
            # No autospec: the mock is not a descriptor, so ``self`` is not
            # prepended - the first positional arg is the command name.
            command = args[0] if args else None
            if command == "CLUSTER SLOTS":
                return init_slots
            if command == "COMMAND":
                return {"get": [], "set": []}
            if command == "INFO":
                return {"cluster_enabled": True}
            if len(args) > 1 and args[1] == "cluster-require-full-coverage":
                return {"cluster-require-full-coverage": "yes"}
            raise AssertionError(f"unexpected command during initialization: {args}")

        execute_command_mock.side_effect = execute_command

        with mock.patch.object(
            AsyncCommandsParser, "initialize", autospec=True
        ) as cmd_parser_initialize:

            def cmd_init_mock(self: AsyncCommandsParser, r: ClusterNode) -> None:
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
            return await MockedRedisCluster(host=HOST, port=7000, **kwargs)


@pytest.mark.fixed_client
async def test_two_moved_one_ask_then_response():
    """
    Script for a GET on slot 12182:
      7000 -> MOVED 12182 7001
      7001 -> MOVED 12182 7002
      7002 -> ASK   12182 7003  (ASKING, then GET)
      7003 -> b"moved-asked-value"

    The final value must be returned and the report must list two MOVED hops
    and one ASK hop, in order, with the correct slot/node transitions.
    """
    client = await _make_client(password=SECRET_PASSWORD)
    slot = 12182

    asking_calls = []

    async def side_effect(self: ClusterNode, *args: Any, **options: Any) -> Any:
        command = args[0]
        # The ASK hop must first issue ASKING on the redirected node.
        if command == "ASKING":
            asking_calls.append(self.port)
            assert self.port == 7003
            return "OK"
        assert command == "GET"
        if self.port == 7000:
            raise _moved(slot, 7001)
        if self.port == 7001:
            raise _moved(slot, 7002)
        if self.port == 7002:
            raise _ask(slot, 7003)
        assert self.port == 7003
        return b"moved-asked-value"

    tracker = RedirectionTracker()
    with mock.patch.object(
        ClusterNode,
        "execute_command",
        autospec=True,
        side_effect=side_effect,
    ):
        result = await client.execute_command("GET", "foo", redirect_tracker=tracker)

    await client.aclose()

    assert result == b"moved-asked-value"
    assert asking_calls == [7003]

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
        redirect_loop=False,
    )
    assert report.hops[1] == RedirectionHop(
        type=RedirectionType.MOVED,
        slot_id=slot,
        from_node=_node(7001),
        to_node=_node(7002),
        redirect_loop=False,
    )
    assert report.hops[2] == RedirectionHop(
        type=RedirectionType.ASK,
        slot_id=slot,
        from_node=_node(7002),
        to_node=_node(7003),
        redirect_loop=False,
    )
    # MOVED patched the slots cache along the way; the ASK hop must not.
    assert client.nodes_manager.get_node_from_slot(slot).name == _node(7002)

    print("\n--- two MOVED + one ASK report ---")
    print(report.summary())


@pytest.mark.fixed_client
async def test_circular_moved_is_flagged_and_raises():
    """
    Script for a GET on slot 12182: MOVED ping-pongs between 7000 and 7001
      7000 -> MOVED 7001, 7001 -> MOVED 7000, ... until TTL is exhausted.

    The request must fail with the TTL-exhausted ClusterError, the report
    must contain only MOVED hops (never ASK), and the repeated edge must be
    flagged as a redirect loop.
    """
    client = await _make_client()
    client.RedisClusterRequestTTL = 4
    slot = 12182

    async def side_effect(self: ClusterNode, *args: Any, **options: Any) -> Any:
        assert args[0] == "GET"
        target = 7001 if self.port == 7000 else 7000
        raise _moved(slot, target)

    tracker = RedirectionTracker()
    with mock.patch.object(
        ClusterNode,
        "execute_command",
        autospec=True,
        side_effect=side_effect,
    ):
        with pytest.raises(ClusterError, match="TTL exhausted"):
            await client.execute_command("GET", "foo", redirect_tracker=tracker)

    await client.aclose()

    report = tracker.report
    assert report.final_node is None
    assert report.ask_count == 0
    assert report.moved_count == 4
    assert all(hop.type is RedirectionType.MOVED for hop in report.hops)
    assert report.redirect_loop_detected is True
    # The first two edges establish the cycle; it repeats from the third hop.
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

    print("\n--- circular MOVED report ---")
    print(report.summary())


@pytest.mark.fixed_client
async def test_ask_does_not_count_as_moved_or_loop():
    """An ASK redirection stays ASK in the report and never flags a loop."""
    client = await _make_client()
    slot = 12182

    async def side_effect(self: ClusterNode, *args: Any, **options: Any) -> Any:
        if args[0] == "ASKING":
            return "OK"
        assert args[0] == "GET"
        if self.port == 7000:
            raise _ask(slot, 7001)
        return b"ask-value"

    tracker = RedirectionTracker()
    with mock.patch.object(
        ClusterNode,
        "execute_command",
        autospec=True,
        side_effect=side_effect,
    ):
        result = await client.execute_command("GET", "foo", redirect_tracker=tracker)

    await client.aclose()

    assert result == b"ask-value"
    report = tracker.report
    assert report.moved_count == 0
    assert report.ask_count == 1
    assert report.redirect_loop_detected is False
    assert report.final_node == _node(7001)


@pytest.mark.fixed_client
async def test_report_contains_no_authentication_information():
    """The tracker only records routing data; secrets must not leak into it."""
    client = await _make_client(password=SECRET_PASSWORD, username="admin")
    slot = client.keyslot(SECRET_PASSWORD)

    async def side_effect(self: ClusterNode, *args: Any, **options: Any) -> Any:
        if args[0] == "ASKING":
            return "OK"
        if self.port == 7000:
            raise _moved(slot, 7001)
        if self.port == 7001:
            raise _ask(slot, 7002)
        return b"safe-value"

    tracker = RedirectionTracker()
    with mock.patch.object(
        ClusterNode,
        "execute_command",
        autospec=True,
        side_effect=side_effect,
    ):
        # The secret rides both as connection auth and as a command argument.
        await client.execute_command("GET", SECRET_PASSWORD, redirect_tracker=tracker)

    await client.aclose()

    rendered = tracker.report.summary()
    assert SECRET_PASSWORD not in rendered
    assert "admin" not in rendered
    assert "password" not in rendered.lower()


@pytest.mark.fixed_client
async def test_invalid_tracker_type_rejected():
    """A non-tracker object must be rejected instead of sent over the wire."""
    client = await _make_client()
    with pytest.raises(TypeError, match="RedirectionTracker"):
        await client.execute_command("GET", "foo", redirect_tracker=object())
    await client.aclose()
