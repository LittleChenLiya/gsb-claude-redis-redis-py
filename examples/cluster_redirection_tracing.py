"""
Minimal, self-contained example of per-request cluster MOVED/ASK redirection
tracing.

It runs entirely against mocked transports - no Redis server, Docker or
network listener is required - and demonstrates both the sync and async
cluster clients:

  1. a request that follows two MOVED redirections and one ASK redirection,
  2. a request stuck in a circular MOVED loop (TTL exhausted).

Run it with::

    python examples/cluster_redirection_tracing.py
"""

import asyncio
from unittest import mock

from redis import Redis
from redis._parsers import AsyncCommandsParser, CommandsParser
from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from redis.exceptions import AskError, ClusterError, MovedError
from redis.redirection import RedirectionTracker

HOST = "127.0.0.1"

# 7000 owns every slot; 7001-7003 are known members so redirects resolve.
CLUSTER_SLOTS = [
    [0, 4095, [HOST, 7000, "n0"], [HOST, 7001, "n1"]],
    [4096, 8191, [HOST, 7000, "n0"], [HOST, 7002, "n2"]],
    [8192, 12287, [HOST, 7000, "n0"], [HOST, 7003, "n3"]],
    [12288, 16383, [HOST, 7000, "n0"], [HOST, 7001, "n1"]],
]

GET_COMMANDS = {
    "get": {
        "name": "get",
        "arity": 2,
        "flags": ["readonly", "fast"],
        "first_key_pos": 1,
        "last_key_pos": 1,
        "step_count": 1,
    }
}


def _moved(slot: int, port: int) -> MovedError:
    return MovedError(f"{slot} {HOST}:{port}")


def _ask(slot: int, port: int) -> AskError:
    return AskError(f"{slot} {HOST}:{port}")


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def _make_sync_client() -> RedisCluster:
    def init_execute_command(*args, **options):
        command = args[0] if args else None
        if command == "CLUSTER SLOTS":
            return CLUSTER_SLOTS
        if command == "COMMAND":
            return {"get": [], "set": []}
        if command == "INFO":
            return {"cluster_enabled": True}
        if len(args) > 1 and args[1] == "cluster-require-full-coverage":
            return {"cluster-require-full-coverage": "yes"}
        raise AssertionError(f"unexpected init command: {args}")

    with (
        mock.patch.object(Redis, "execute_command", side_effect=init_execute_command),
        mock.patch.object(CommandsParser, "initialize", autospec=True) as cmd_init,
    ):

        def init_parser(self, r):
            self.commands = GET_COMMANDS

        cmd_init.side_effect = init_parser

        class MockedRedisCluster(RedisCluster):
            def __del__(self):
                pass

        client = MockedRedisCluster(host=HOST, port=7000, reinitialize_steps=0)

    for node in client.get_nodes():
        connection = mock.Mock()
        connection.host = node.host
        connection.port = node.port
        connection.is_connected = True
        node.redis_connection.connection = connection

    return client


def run_sync() -> None:
    slot = 12182

    # Script 1: two MOVED, one ASK, then the final response.
    client = _make_sync_client()

    def happy_script(connection, command_name):
        if command_name == "ASKING":
            return "OK"
        if connection.port == 7000:
            raise _moved(slot, 7001)
        if connection.port == 7001:
            raise _moved(slot, 7002)
        if connection.port == 7002:
            raise _ask(slot, 7003)
        return b"moved-asked-value"

    tracker = RedirectionTracker()
    parse_response = mock.Mock(side_effect=happy_script)
    with mock.patch.object(Redis, "parse_response", parse_response):
        result = client.execute_command("GET", "foo", redirect_tracker=tracker)
    print("sync final response:", result)
    print(tracker.report.summary())
    print()
    client.close()

    # Script 2: circular MOVED loop.
    client = _make_sync_client()
    client.RedisClusterRequestTTL = 4

    def loop_script(connection, command_name):
        target = 7001 if connection.port == 7000 else 7000
        raise _moved(slot, target)

    tracker = RedirectionTracker()
    parse_response = mock.Mock(side_effect=loop_script)
    with mock.patch.object(Redis, "parse_response", parse_response):
        try:
            client.execute_command("GET", "foo", redirect_tracker=tracker)
        except ClusterError as error:
            print("sync circular redirect raised:", error)
    print(tracker.report.summary())
    client.close()


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


async def _make_async_client() -> AsyncRedisCluster:
    with mock.patch.object(AsyncClusterNode, "execute_command") as execute_command_mock:

        async def init_execute_command(*args, **options):
            command = args[0] if args else None
            if command == "CLUSTER SLOTS":
                return CLUSTER_SLOTS
            if command == "COMMAND":
                return {"get": [], "set": []}
            if command == "INFO":
                return {"cluster_enabled": True}
            if len(args) > 1 and args[1] == "cluster-require-full-coverage":
                return {"cluster-require-full-coverage": "yes"}
            raise AssertionError(f"unexpected init command: {args}")

        execute_command_mock.side_effect = init_execute_command

        with mock.patch.object(
            AsyncCommandsParser, "initialize", autospec=True
        ) as cmd_init:

            def init_parser(self, r):
                self.commands = GET_COMMANDS

            cmd_init.side_effect = init_parser

            class MockedAsyncRedisCluster(AsyncRedisCluster):
                def __del__(self):
                    pass

            return await MockedAsyncRedisCluster(
                host=HOST, port=7000, reinitialize_steps=0
            )


async def run_async() -> None:
    slot = 12182

    # Script 1: two MOVED, one ASK, then the final response.
    client = await _make_async_client()

    async def happy_side_effect(node, *args, **options):
        command = args[0]
        if command == "ASKING":
            return "OK"
        if node.port == 7000:
            raise _moved(slot, 7001)
        if node.port == 7001:
            raise _moved(slot, 7002)
        if node.port == 7002:
            raise _ask(slot, 7003)
        return b"moved-asked-value"

    tracker = RedirectionTracker()
    with mock.patch.object(
        AsyncClusterNode,
        "execute_command",
        autospec=True,
        side_effect=happy_side_effect,
    ):
        result = await client.execute_command("GET", "foo", redirect_tracker=tracker)
    print("async final response:", result)
    print(tracker.report.summary())
    print()
    await client.aclose()

    # Script 2: circular MOVED loop.
    client = await _make_async_client()
    client.RedisClusterRequestTTL = 4

    async def loop_side_effect(node, *args, **options):
        target_port = 7001 if node.port == 7000 else 7000
        raise _moved(slot, target_port)

    tracker = RedirectionTracker()
    with mock.patch.object(
        AsyncClusterNode,
        "execute_command",
        autospec=True,
        side_effect=loop_side_effect,
    ):
        try:
            await client.execute_command("GET", "foo", redirect_tracker=tracker)
        except ClusterError as error:
            print("async circular redirect raised:", error)
    print(tracker.report.summary())
    await client.aclose()


def main() -> None:
    print("=" * 60)
    print("SYNC RedisCluster")
    print("=" * 60)
    run_sync()
    print()
    print("=" * 60)
    print("ASYNC RedisCluster")
    print("=" * 60)
    asyncio.run(run_async())
    print()
    print(
        "No authentication data is recorded: the tracker stores only the"
        " redirect kind, slot id and host:port node names."
    )


if __name__ == "__main__":
    main()
