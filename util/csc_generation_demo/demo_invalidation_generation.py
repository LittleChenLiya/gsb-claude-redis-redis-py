"""
Reproducible end-to-end demonstration (native, no Docker):

    python util/csc_generation_demo/demo_invalidation_generation.py

Starts the in-repo minimal RESP3 server (fake_redis_server.py), then drives a
real redis.Redis client with client-side caching enabled through:

    read -> disconnect -> reconnect -> out-of-order invalidations

At every step it prints the returned value, the number of GET commands that
actually reached the wire (a cache hit adds none), the cache size and the
invalidation generations.

What it proves:
  1. the first read is learned on the wire; the next is a cache hit;
  2. disconnect flushes the cache and bumps the cache-wide generation; reconnect
     bumps the per-connection generation and installs a new push handler;
  3. a value re-learned on the new socket is untouched by an invalidation
     replayed by the *previous* (late) handler;
  4. a late flush-invalidation (invalidate, null) is dropped too;
  5. an invalidation delivered by the *current* connection still invalidates.
"""

import sys
import time

from fake_redis_server import FakeRedisServer

from redis import Redis
from redis.cache import CacheConfig, CacheKey, DefaultCache

GET_FOO_KEY = CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))


def headline(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main():
    server = FakeRedisServer()
    cache = DefaultCache(CacheConfig(max_size=10))

    # protocol=3 is required for server-assisted client-side caching.
    client = Redis(
        host=server.host,
        port=server.port,
        protocol=3,
        cache=cache,
        socket_connect_timeout=5,
        socket_timeout=5,
    )

    try:
        headline("1. READ: learn on the wire, then serve from cache")
        first = client.get("foo")
        wire_after_first = server.get_count
        second = client.get("foo")  # cache hit: no wire GET
        print(f"first  GET foo -> {first!r}; wire GETs so far = {wire_after_first}")
        print(
            f"second GET foo -> {second!r} (cache hit); wire GETs = {server.get_count}"
        )
        assert first == second == b"v1"
        assert server.get_count == wire_after_first, (
            "second read must not touch the wire"
        )

        pool = client.connection_pool
        proxy_conn = pool.get_connection()
        print(
            f"cache size = {cache.size}; cache generation = "
            f"{cache.invalidation_generation}; connection generation = "
            f"{proxy_conn._invalidation_generation}"
        )
        assert cache.size == 1
        assert cache.invalidation_generation == 0
        assert proxy_conn._invalidation_generation == 1
        pool.release(proxy_conn)

        headline("2. DISCONNECT + RECONNECT: generations advance, old value is gone")
        proxy_conn = pool.get_connection()
        old_handler = proxy_conn._conn._parser.invalidation_push_handler_func
        proxy_conn.disconnect()
        print(
            f"after disconnect : cache size = {cache.size}; cache generation = "
            f"{cache.invalidation_generation}; connection generation = "
            f"{proxy_conn._invalidation_generation}"
        )
        assert cache.size == 0
        assert cache.invalidation_generation == 1
        assert cache.get(GET_FOO_KEY) is None
        assert proxy_conn._invalidation_generation == 2  # 1st connect=1, disconnect=2

        # The server-side value changes while the client is disconnected.
        server.values[b"foo"] = b"v2"
        proxy_conn.connect()
        new_handler = proxy_conn._conn._parser.invalidation_push_handler_func
        print(
            f"after reconnect  : connection generation = "
            f"{proxy_conn._invalidation_generation}; fresh handler installed = "
            f"{new_handler is not old_handler}; server connections accepted = "
            f"{server.connections}"
        )
        assert proxy_conn._invalidation_generation == 3  # reconnect bumps again
        assert new_handler is not old_handler
        assert server.connections == 2
        pool.release(proxy_conn)

        headline("3. READ on the new socket: a hit cannot cross generations")
        value = client.get("foo")  # must re-learn on the wire
        print(f"GET foo -> {value!r} (re-learned); wire GETs = {server.get_count}")
        assert value == b"v2"
        entry = cache.get(GET_FOO_KEY)
        assert entry is not None and entry.generation == cache.invalidation_generation
        hit = client.get("foo")  # cache hit
        print(f"GET foo -> {hit!r} (cache hit); wire GETs = {server.get_count}")
        assert hit == b"v2"
        assert entry.cache_value == b"v2"

        headline("4. OUT OF ORDER: late invalidation from the PREVIOUS connection")
        # Message the old socket pushed, but whose handling was delayed until after
        # the reconnect (bytes left unread). Its handler closure is bound to the
        # previous connection generation and must be dropped.
        old_handler([b"invalidate", [b"foo"]])
        print("replayed stale [invalidate, [foo]]  -> must be dropped")
        assert cache.size == 1
        assert cache.get(GET_FOO_KEY).cache_value == b"v2"
        stale_after_key = client.get("foo")
        print(
            f"GET foo -> {stale_after_key!r} (still a cache hit); wire GETs = {server.get_count}"
        )
        assert stale_after_key == b"v2"

        old_handler([b"invalidate", None])
        print("replayed stale [invalidate, null]   -> must be dropped (no flush)")
        assert cache.size == 1

        headline("5. INVALIDATION ON THE CURRENT CONNECTION still takes effect")
        # A push genuinely delivered by the live socket: the next command drains
        # pending pushes before it would serve the cached value.
        server.invalidate([b"foo"])
        deadline = time.time() + 3
        while cache.size == 1 and time.time() < deadline:
            client.get("foo")  # drains the push; re-learns if invalidated
        print(
            f"after live push: cache size = {cache.size}; wire GETs = {server.get_count}"
        )
        assert cache.get(GET_FOO_KEY) is not None
        assert client.get("foo") == b"v2"

        headline("ALL CHECKS PASSED")
        print(
            f"final: wire GETs = {server.get_count}; cache size = {cache.size}; "
            f"cache generation = {cache.invalidation_generation}; connection "
            f"generation = {proxy_conn._invalidation_generation}"
        )
    finally:
        client.close()
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
