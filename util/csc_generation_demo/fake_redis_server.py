"""
A tiny, dependency-free RESP3 server, just sufficient to demonstrate the
client-side caching invalidation-generation machinery without Docker or a real
Redis binary.

It supports one client connection at a time and implements:
  * the RESP3 HELLO handshake (advertising redis 7.4.0);
  * CLIENT SETINFO / CLIENT TRACKING ON / PING (replies OK/PONG);
  * GET <key> from an in-memory dict;
  * server-initiated RESP3 "invalidate" push messages.

The number of GET commands that reached the wire is exposed so the demo can
tell a cache hit (no wire GET) from a miss apart.
"""

import socket
import threading


def enc_simple(s: bytes) -> bytes:
    return b"+" + s + b"\r\n"


def enc_int(n: int) -> bytes:
    return b":" + str(n).encode() + b"\r\n"


def enc_bulk(b) -> bytes:
    if b is None:
        return b"$-1\r\n"
    return b"$" + str(len(b)).encode() + b"\r\n" + b + b"\r\n"


def enc_map(pairs) -> bytes:
    out = b"%" + str(len(pairs)).encode() + b"\r\n"
    for k, v in pairs:
        out += enc_bulk(k)
        out += v
    return out


def enc_push(items) -> bytes:
    # RESP3 push: >N\r\n followed by bulk strings (or null arrays).
    out = b">" + str(len(items)).encode() + b"\r\n"
    for item in items:
        if item is None:
            out += b"_\r\n"  # RESP3 null
        elif isinstance(item, list):
            if not item:
                out += b"*0\r\n"
            else:
                out += b"*" + str(len(item)).encode() + b"\r\n"
                out += b"".join(enc_bulk(x) for x in item)
        else:
            out += enc_bulk(item)
    return out


class _Reader:
    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def _need(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("client closed")
            self.buf += chunk

    def read_line(self):
        while b"\r\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("client closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def read_exact(self, n):
        self._need(n + 2)
        data = self.buf[:n]
        self.buf = self.buf[n + 2 :]
        return data

    def read_command(self):
        line = self.read_line()
        if not line.startswith(b"*"):
            raise ValueError(f"unexpected: {line!r}")
        argc = int(line[1:])
        args = []
        for _ in range(argc):
            bulk_line = self.read_line()
            if not bulk_line.startswith(b"$"):
                raise ValueError(f"unexpected: {bulk_line!r}")
            length = int(bulk_line[1:])
            args.append(self.read_exact(length))
        return args


class FakeRedisServer:
    def __init__(self, host="127.0.0.1", port=0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.host, self.port = self.sock.getsockname()
        self.values = {b"foo": b"v1"}
        self.get_count = 0
        self.connections = 0
        self._conn_sock = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            self._conn_sock = conn
            try:
                self._serve(conn)
            except ConnectionError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _serve(self, conn):
        reader = _Reader(conn)

        def send(data):
            conn.sendall(data)

        while True:
            args = reader.read_command()
            cmd = args[0].upper()
            if cmd == b"HELLO":
                send(
                    enc_map(
                        [
                            (b"server", enc_bulk(b"redis")),
                            (b"version", enc_bulk(b"7.4.0")),
                            (b"proto", enc_int(3)),
                            (b"id", enc_int(1)),
                            (b"mode", enc_bulk(b"standalone")),
                            (b"role", enc_bulk(b"master")),
                        ]
                    )
                )
            elif cmd == b"CLIENT" and len(args) > 1 and args[1].upper() == b"TRACKING":
                send(enc_simple(b"OK"))
            elif cmd == b"CLIENT":
                send(enc_simple(b"OK"))  # SETNAME / SETINFO
            elif cmd == b"PING":
                send(enc_simple(b"PONG"))
            elif cmd == b"GET":
                self.get_count += 1
                send(enc_bulk(self.values.get(args[1])))
            elif cmd == b"SELECT":
                send(enc_simple(b"OK"))
            else:
                send(enc_simple(b"OK"))

    def invalidate(self, keys):
        """Server-pushed invalidation for the live connection (out-of-order-able)."""
        items = [b"invalidate"]
        items.append(
            None
            if keys is None
            else [k.encode() if isinstance(k, str) else k for k in keys]
        )
        self._conn_sock.sendall(enc_push(items))

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
