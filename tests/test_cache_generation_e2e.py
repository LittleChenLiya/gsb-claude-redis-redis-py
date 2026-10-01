# ruff: noqa: E402, I001
"""
End-to-end, dependency-free tests for the client-side caching invalidation
generation.

These tests need neither a real Redis binary nor Docker: they spin up the
in-repo minimal RESP3 server (``util/csc_generation_demo/fake_redis_server.py``)
on a localhost socket and drive a real synchronous ``redis.Redis`` client with
caching enabled through read / disconnect / reconnect / out-of-order
invalidations, asserting on the returned values, wire traffic and generations.

Run:
    pytest tests/test_cache_generation_e2e.py -v
"""

import os
import sys
import time

import pytest

from redis import Redis
from redis.cache import CacheConfig, CacheKey, DefaultCache

_UTIL = os.path.join(os.path.dirname(__file__), "..", "util", "csc_generation_demo")
sys.path.insert(0, os.path.abspath(_UTIL))

from fake_redis_server import FakeRedisServer  # noqa: E402  # sys.path set just above

GET_FOO = CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))


@pytest.fixture()
def fake_server():
    server = FakeRedisServer()
    yield server
    server.stop()


@pytest.fixture()
def cached_client(fake_server):
    cache = DefaultCache(CacheConfig(max_size=10))
    client = Redis(
        host=fake_server.host,
        port=fake_server.port,
        protocol=3,
        cache=cache,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    yield client, cache
    client.close()


def test_read_disconnect_reconnect_out_of_order_invalidations(
    fake_server, cached_client
):
    """The full acceptance scenario over a real TCP socket."""
    client, cache = cached_client
    pool = client.connection_pool

    # --- 1. read: learn on the wire, then serve from cache ---------------
    assert client.get("foo") == b"v1"
    wire_gets = fake_server.get_count
    assert wire_gets == 1
    assert client.get("foo") == b"v1"  # cache hit
    assert fake_server.get_count == wire_gets
    assert cache.size == 1
    assert cache.invalidation_generation == 0
    entry = cache.get(GET_FOO)
    assert entry.generation == 0

    # --- 2. disconnect + reconnect: generations advance, cache flushed --
    proxy = pool.get_connection()
    assert proxy._invalidation_generation == 1
    old_handler = proxy._conn._parser.invalidation_push_handler_func

    proxy.disconnect()
    assert cache.size == 0
    assert cache.get(GET_FOO) is None
    assert cache.invalidation_generation == 1
    assert proxy._invalidation_generation == 2

    fake_server.values[b"foo"] = b"v2"
    proxy.connect()
    assert proxy._invalidation_generation == 3
    assert proxy._conn._parser.invalidation_push_handler_func is not old_handler
    assert fake_server.connections == 2
    pool.release(proxy)

    # --- 3. read on the new socket: no cross-generation hit --------------
    assert client.get("foo") == b"v2"  # miss, re-learned from wire
    assert fake_server.get_count == wire_gets + 1
    assert cache.get(GET_FOO).generation == 1
    assert cache.get(GET_FOO).cache_value == b"v2"
    assert client.get("foo") == b"v2"  # hit
    assert fake_server.get_count == wire_gets + 1

    # --- 4. late invalidation replayed by the previous handler ----------
    old_handler([b"invalidate", [b"foo"]])
    assert cache.size == 1
    assert cache.get(GET_FOO).cache_value == b"v2"
    assert client.get("foo") == b"v2"  # still a hit, no wire GET
    assert fake_server.get_count == wire_gets + 1

    # A late whole-cache flush is discarded as well.
    old_handler([b"invalidate", None])
    assert cache.size == 1

    # --- 5. invalidation on the live connection still applies -----------
    fake_server.invalidate([b"foo"])
    deadline = time.time() + 3
    while cache.size == 1 and time.time() < deadline:
        client.get("foo")  # drains the pending push, re-learns if invalidated
    assert fake_server.get_count == wire_gets + 2
    assert client.get("foo") == b"v2"


def test_live_flush_invalidation_clears_the_cache(fake_server, cached_client):
    """data[1] is None from the current generation -> the cache is flushed.

    The draining GET then misses and re-learns the value, so the observable
    effect is an extra GET reaching the wire (the previous read was a hit), not a
    permanently empty cache.
    """
    client, cache = cached_client
    assert client.get("foo") == b"v1"
    assert cache.size == 1
    assert client.get("foo") == b"v1"  # hit, no wire traffic
    wire_gets = fake_server.get_count

    fake_server.invalidate(None)
    deadline = time.time() + 3
    while fake_server.get_count == wire_gets and time.time() < deadline:
        client.get("foo")  # drains the pending flush and re-learns on a miss
    assert fake_server.get_count == wire_gets + 1
    assert client.get("foo") == b"v1"
