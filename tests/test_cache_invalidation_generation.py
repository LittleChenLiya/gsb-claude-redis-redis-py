"""
Tests for client-side cache invalidation generation (代次).

A cache generation is advanced on every disconnect/reconnect cycle. Entries created in
a previous generation are stale: a lookup must never serve them as a cache hit, and an
invalidation push produced for a previous tracking session must be discarded as late.

These tests use a small in-process RESP3 server (no Docker, no external Redis binary) so
the sequence read -> disconnect -> reconnect -> out-of-order invalidation is fully
controllable and reproducible.
"""

import socket
import threading
from contextlib import contextmanager

import redis

from redis.cache import (
    CacheConfig,
    CacheEntry,
    CacheEntryStatus,
    CacheKey,
    DefaultCache,
)


class TestCacheGeneration:
    def test_starts_at_generation_zero(self):
        cache = DefaultCache(CacheConfig(max_size=10))
        assert cache.generation == 0

    def test_next_generation_advances_monotonically(self):
        cache = DefaultCache(CacheConfig(max_size=10))
        assert cache.next_generation() == 1
        assert cache.next_generation() == 2
        assert cache.next_generation() == 3
        assert cache.generation == 3

    def test_set_tags_entry_with_current_generation(self):
        cache = DefaultCache(CacheConfig(max_size=10))
        key = CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
        entry = CacheEntry(
            cache_key=key,
            cache_value=b"bar",
            status=CacheEntryStatus.VALID,
            connection_ref=object(),
        )
        cache.set(entry)
        assert entry.generation == 0

        cache.next_generation()
        entry2 = CacheEntry(
            cache_key=CacheKey(
                command="GET", redis_keys=("foo2",), redis_args=("GET", "foo2")
            ),
            cache_value=b"baz",
            status=CacheEntryStatus.VALID,
            connection_ref=object(),
        )
        cache.set(entry2)
        assert entry2.generation == 1

    def test_get_rejects_cross_generation_entries(self):
        cache = DefaultCache(CacheConfig(max_size=10))
        key = CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
        entry = CacheEntry(
            cache_key=key,
            cache_value=b"bar",
            status=CacheEntryStatus.VALID,
            connection_ref=object(),
        )
        cache.set(entry)
        assert cache.get(key).cache_value == b"bar"

        cache.next_generation()
        assert cache.get(key) is None
        assert cache.size == 0

    def test_get_returns_current_generation_entries(self):
        cache = DefaultCache(CacheConfig(max_size=10))
        key = CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
        entry = CacheEntry(
            cache_key=key,
            cache_value=b"bar",
            status=CacheEntryStatus.VALID,
            connection_ref=object(),
        )
        cache.set(entry)
        assert cache.get(key) is not None

        cache.next_generation()
        new_entry = CacheEntry(
            cache_key=key,
            cache_value=b"bar2",
            status=CacheEntryStatus.VALID,
            connection_ref=object(),
        )
        cache.set(new_entry)
        assert cache.get(key).cache_value == b"bar2"

    def test_flush_resets_entries_but_not_generation(self):
        cache = DefaultCache(CacheConfig(max_size=10))
        key = CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
        cache.set(
            CacheEntry(
                cache_key=key,
                cache_value=b"bar",
                status=CacheEntryStatus.VALID,
                connection_ref=object(),
            )
        )
        cache.next_generation()
        assert cache.flush() == 1
        assert cache.size == 0
        assert cache.generation == 1


class _FakeResp3Server:
    """
    A minimal RESP3 server supporting HELLO, CLIENT TRACKING, GET, SET, PING.

    Sends invalidation pushes to every tracking connection when a key changes.
    The test drives read -> disconnect -> reconnect -> out-of-order invalidation.
    """

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._threads = []
        self._tracking_conns = set()
        self._keys = {b"foo": b"bar"}
        self._lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        for t in self._threads:
            t.join(timeout=1)

    @staticmethod
    def _read_line(conn):
        buf = b""
        while not buf.endswith(b"\r\n"):
            chunk = conn.recv(1)
            if not chunk:
                raise ConnectionError("connection closed")
            buf += chunk
        return buf[:-2]

    def _read_resp(self, conn):
        line = self._read_line(conn)
        prefix, rest = line[:1], line[1:]
        if prefix == b"*":
            count = int(rest)
            return [self._read_resp(conn) for _ in range(count)]
        if prefix == b"$":
            length = int(rest)
            if length == -1:
                return None
            data = b""
            while len(data) < length + 2:
                chunk = conn.recv(length + 2 - len(data))
                if not chunk:
                    raise ConnectionError("connection closed")
                data += chunk
            return data[:length]
        if prefix == b":":
            return int(rest)
        return line.split()

    @staticmethod
    def _send(conn, payload):
        conn.sendall(payload)

    @staticmethod
    def _bulk(value):
        return b"$%d\r\n%s\r\n" % (len(value), value)

    def _hello_response(self):
        pairs = [
            (b"server", b"redis"),
            (b"version", b"7.4.0"),
            (b"proto", b"3"),
            (b"id", b"1"),
            (b"mode", b"standalone"),
            (b"role", b"master"),
        ]
        out = b"%" + str(len(pairs)).encode() + b"\r\n"
        for key, value in pairs:
            out += self._bulk(key)
            if key == b"proto" or key == b"id":
                out += b":" + value + b"\r\n"
            else:
                out += self._bulk(value)
        return out

    def _invalidate(self, key, except_conn=None):
        msg = b">2\r\n$10\r\ninvalidate\r\n*1\r\n" + self._bulk(key)
        dead = []
        for conn in list(self._tracking_conns):
            if conn is except_conn:
                continue
            try:
                self._send(conn, msg)
            except OSError:
                dead.append(conn)
        for conn in dead:
            self._tracking_conns.discard(conn)

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                self._sock.settimeout(0.2)
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            t.start()
            self._threads.append(t)

    def _handle(self, conn):
        try:
            while not self._stop.is_set():
                try:
                    cmd = self._read_resp(conn)
                except (ConnectionError, OSError):
                    break
                if not cmd:
                    break
                name = cmd[0].upper()
                if name == b"HELLO":
                    self._send(conn, self._hello_response())
                elif name == b"CLIENT" and len(cmd) > 1 and cmd[1].upper() == b"TRACKING":
                    self._tracking_conns.add(conn)
                    self._send(conn, b"+OK\r\n")
                elif name == b"GET":
                    key = cmd[1]
                    with self._lock:
                        value = self._keys.get(key)
                    if value is None:
                        self._send(conn, b"_\r\n")
                    else:
                        self._send(conn, self._bulk(value))
                elif name == b"SET":
                    key, value = cmd[1], cmd[2]
                    with self._lock:
                        self._keys[key] = value
                    self._send(conn, b"+OK\r\n")
                    self._invalidate(key, except_conn=conn)
                elif name == b"PING":
                    self._send(conn, b"+PONG\r\n")
                else:
                    self._send(conn, b"+OK\r\n")
        except (ConnectionError, OSError):
            pass
        finally:
            self._tracking_conns.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def set_key(self, key, value):
        with self._lock:
            self._keys[key] = value
        self._invalidate(key)


@contextmanager
def fake_redis_server():
    server = _FakeResp3Server()
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _make_client(server, **kwargs):
    return redis.Redis(
        host="127.0.0.1",
        port=server.port,
        protocol=3,
        cache=DefaultCache(CacheConfig(max_size=100)),
        **kwargs,
    )


class TestInvalidationGenerationFlow:
    """
    Simulate read -> disconnect -> reconnect -> out-of-order invalidation.

      1. Read foo -> cached in generation 0.
      2. Disconnect -> generation advances to 1; old entry is stale.
      3. Reconnect -> tracking session restarts at generation 1.
      4. Read foo -> cross-generation miss -> re-cached in generation 1.
      5. Out-of-order invalidation for foo (produced for generation 0) arrives late.
      6. The late invalidation is discarded; the generation-1 value is preserved.
    """

    def test_late_invalidation_discarded_after_reconnect(self):
        with fake_redis_server() as server:
            client = _make_client(server, single_connection_client=True)

            assert client.get("foo") == b"bar"
            cache = client.get_cache()
            assert cache.generation == 0
            assert cache.get(
                CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
            ).cache_value == b"bar"

            client.connection_pool.disconnect()
            assert cache.generation == 1

            assert client.get("foo") == b"bar"
            assert cache.generation == 1
            entry = cache.get(
                CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
            )
            assert entry is not None
            assert entry.generation == 1

            # Deliver an out-of-order invalidation produced for the old tracking
            # session (generation 0). The connection's tracking generation is 1, so
            # this push is late and must be discarded without evicting the value.
            proxy = client.connection_pool.get_connection()
            assert proxy._tracking_generation == 1
            proxy._tracking_generation = 0
            proxy._on_invalidation_callback([b"invalidate", [b"foo"]])
            proxy._tracking_generation = 1

            entry = cache.get(
                CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
            )
            assert entry is not None
            assert entry.cache_value == b"bar"
            assert entry.generation == 1

            client.close()

    def test_current_session_invalidation_still_applies(self):
        """Invalidations for the current tracking session must still evict entries."""
        with fake_redis_server() as server:
            client = _make_client(server, single_connection_client=True)

            assert client.get("foo") == b"bar"
            cache = client.get_cache()
            assert cache.generation == 0

            proxy = client.connection_pool.get_connection()
            assert proxy._tracking_generation == 0
            proxy._on_invalidation_callback([b"invalidate", [b"foo"]])

            assert cache.get(
                CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
            ) is None

            client.close()

    def test_cache_hit_never_crosses_generation(self):
        """After a reconnect, a cached value from the old generation is never served."""
        with fake_redis_server() as server:
            client = _make_client(server, single_connection_client=True)

            assert client.get("foo") == b"bar"
            cache = client.get_cache()

            client.connection_pool.disconnect()
            assert cache.generation == 1

            assert cache.get(
                CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
            ) is None
            assert client.get("foo") == b"bar"
            assert cache.get(
                CacheKey(command="GET", redis_keys=("foo",), redis_args=("GET", "foo"))
            ).generation == 1

            client.close()

    def test_generation_advances_on_each_disconnect(self):
        with fake_redis_server() as server:
            client = _make_client(server, single_connection_client=True)
            cache = client.get_cache()

            assert client.get("foo") == b"bar"
            assert cache.generation == 0

            client.connection_pool.disconnect()
            assert cache.generation == 1

            assert client.get("foo") == b"bar"
            assert cache.generation == 1

            client.connection_pool.disconnect()
            assert cache.generation == 2

            client.close()
