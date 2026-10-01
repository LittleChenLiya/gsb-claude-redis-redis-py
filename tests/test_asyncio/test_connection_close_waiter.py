"""Acceptance tests for the async connection close waiter.

A fake StreamReader/StreamWriter pair drives the read/close/reconnect races
without a real server. The tests assert that:

* a pending read or write fails immediately with the consistent
  SERVER_CLOSED_CONNECTION_ERROR when the connection is closed,
* a reconnected (new-generation) connection is not poisoned by the close
  event of the previous generation,
* no helper task is left pending after the races resolve.
"""

import asyncio

import pytest

from redis._parsers.socket import SERVER_CLOSED_CONNECTION_ERROR
from redis.asyncio.connection import Connection
from redis.exceptions import ConnectionError


class FakeStreamReader:
    """Minimal asyncio.StreamReader stand-in with test-controlled feeding."""

    def __init__(self):
        self._buffer = bytearray()
        self._eof = False
        self._waiter = None

    def feed_data(self, data: bytes):
        self._buffer.extend(data)
        self._wake()

    def feed_eof(self):
        self._eof = True
        self._wake()

    def _wake(self):
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)

    def at_eof(self):
        return self._eof and not self._buffer

    async def readline(self):
        while True:
            idx = self._buffer.find(b"\n")
            if idx >= 0:
                line = bytes(self._buffer[: idx + 1])
                del self._buffer[: idx + 1]
                return line
            if self._eof:
                return b""
            self._waiter = asyncio.get_running_loop().create_future()
            try:
                await self._waiter
            finally:
                self._waiter = None

    async def readexactly(self, n):
        while len(self._buffer) < n:
            if self._eof:
                raise asyncio.IncompleteReadError(bytes(self._buffer), n)
            self._waiter = asyncio.get_running_loop().create_future()
            try:
                await self._waiter
            finally:
                self._waiter = None
        data = bytes(self._buffer[:n])
        del self._buffer[:n]
        return data


class FakeTransport:
    def __init__(self):
        self._closing = False

    def is_closing(self):
        return self._closing

    def get_extra_info(self, name, default=None):
        if name == "peername":
            return ("127.0.0.1", 6379)
        if name == "sockname":
            return ("127.0.0.1", 12345)
        return default


class FakeStreamWriter:
    """Minimal asyncio.StreamWriter stand-in; drain() can be made to block."""

    def __init__(self, block_drain: bool = False):
        self.transport = FakeTransport()
        self._block_drain = block_drain
        self.written = bytearray()
        self.closed = False

    def writelines(self, chunks):
        for chunk in chunks:
            self.written.extend(chunk)

    def write(self, data):
        self.written.extend(data)

    async def drain(self):
        if self.closed:
            # Like a real transport: writing to a closed connection fails.
            raise ConnectionResetError("Connection closed")
        if self._block_drain:
            # Block forever: only the close waiter (or the test) unblocks us.
            await asyncio.get_running_loop().create_future()

    def close(self):
        self.closed = True
        self.transport._closing = True

    async def wait_closed(self):
        return None

    def get_extra_info(self, name, default=None):
        return self.transport.get_extra_info(name, default)


async def connect_with(conn, reader, writer):
    """Drive the real connect path with a fake transport pair."""

    async def fake__connect():
        conn._reader = reader
        conn._writer = writer

    conn._connect = fake__connect
    # Skip the handshake commands but keep the parser wiring that
    # on_connect_check_health() would normally perform.
    conn.redis_connect_func = lambda c: c._parser.on_connect(c)
    await conn.connect_check_health()
    return conn


def make_connection(**kwargs):
    kwargs.setdefault("socket_timeout", None)
    kwargs.setdefault("socket_connect_timeout", None)
    return Connection(**kwargs)


def leftover_tasks():
    current = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not current and not t.done()]


async def test_pending_read_fails_fast_on_disconnect():
    conn = make_connection()
    await connect_with(conn, FakeStreamReader(), FakeStreamWriter())
    assert conn.is_connected

    read_task = asyncio.create_task(conn.read_response())
    await asyncio.sleep(0)  # let the read block on the empty fake reader
    assert not read_task.done()

    await asyncio.wait_for(conn.disconnect(), timeout=1)
    with pytest.raises(ConnectionError, match="Connection closed by server"):
        await asyncio.wait_for(read_task, timeout=1)

    await asyncio.sleep(0)
    assert leftover_tasks() == []


async def test_pending_write_fails_fast_on_disconnect():
    conn = make_connection()
    await connect_with(conn, FakeStreamReader(), FakeStreamWriter(block_drain=True))

    send_task = asyncio.create_task(conn.send_command("PING"))
    await asyncio.sleep(0)  # let the write block in drain()
    assert not send_task.done()

    await asyncio.wait_for(conn.disconnect(), timeout=1)
    with pytest.raises(ConnectionError, match="Connection closed by server"):
        await asyncio.wait_for(send_task, timeout=1)

    await asyncio.sleep(0)
    assert leftover_tasks() == []


async def test_concurrent_waiters_get_consistent_error_and_no_task_leaks():
    conn = make_connection()
    await connect_with(conn, FakeStreamReader(), FakeStreamWriter(block_drain=True))

    waiters = [asyncio.create_task(conn.read_response()) for _ in range(3)]
    waiters.append(asyncio.create_task(conn.send_command("PING")))
    await asyncio.sleep(0)
    assert all(not t.done() for t in waiters)

    await conn.disconnect()
    results = await asyncio.gather(*waiters, return_exceptions=True)
    # Every waiter observes the exact same close error.
    for result in results:
        assert isinstance(result, ConnectionError)
        assert str(result) == SERVER_CLOSED_CONNECTION_ERROR

    await asyncio.sleep(0)
    assert leftover_tasks() == []


async def test_reconnect_is_not_poisoned_by_old_close_event():
    conn = make_connection()
    await connect_with(conn, FakeStreamReader(), FakeStreamWriter())
    old_event = conn._close_event

    read_task = asyncio.create_task(conn.read_response())
    await asyncio.sleep(0)
    await conn.disconnect()
    with pytest.raises(ConnectionError, match="Connection closed by server"):
        await read_task
    assert old_event.is_set()

    # Reconnect: a fresh generation must get a fresh, unset close event.
    reader2 = FakeStreamReader()
    await connect_with(conn, reader2, FakeStreamWriter())
    new_event = conn._close_event
    assert new_event is not old_event
    assert not new_event.is_set()

    # The new generation reads normally; the old close does not interfere.
    reader2.feed_data(b"+PONG\r\n")
    response = await asyncio.wait_for(conn.read_response(), timeout=1)
    assert response == b"PONG"

    # Closing the new generation only signals its own event.
    await conn.disconnect()
    assert new_event.is_set()

    await asyncio.sleep(0)
    assert leftover_tasks() == []


async def test_read_after_close_fails_fast_without_reconnect():
    conn = make_connection()
    await connect_with(conn, FakeStreamReader(), FakeStreamWriter())
    await conn.disconnect()

    # A late read on the closed generation gets the same consistent error
    # immediately instead of hanging on the dead stream.
    with pytest.raises(ConnectionError, match="Connection closed by server"):
        await asyncio.wait_for(conn.read_response(), timeout=1)

    await asyncio.sleep(0)
    assert leftover_tasks() == []


async def test_successful_read_leaves_no_pending_tasks():
    conn = make_connection()
    reader = FakeStreamReader()
    await connect_with(conn, reader, FakeStreamWriter())

    reader.feed_data(b"+OK\r\n")
    assert await conn.read_response() == b"OK"

    await conn.disconnect()
    await asyncio.sleep(0)
    assert leftover_tasks() == []
