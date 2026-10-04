"""
Bounded local buffering adapter for PubSub consumers.

This module implements a *local-only* acknowledgement layer on top of a
PubSub-like message source (anything exposing ``get_message(timeout=...)``,
e.g. :class:`redis.client.PubSub`).

Semantics are deliberately local:

* Messages are pulled from the underlying source into a bounded in-memory
  buffer.  When the buffer is full (buffered plus in-flight messages equals
  capacity) the adapter stops reading from the source, providing
  backpressure.  Nothing is persisted to Redis or anywhere else.
* The caller explicitly acks each leased message.  A message that has been
  leased but not yet acked is never released from local tracking and is
  never handed to another consumer.
* ``nack()`` returns the message to the *front* of the local queue and
  increments its local redelivery counter.  Once the counter reaches
  ``max_redeliveries`` the message is dropped and the caller's
  ``on_failure`` callback is invoked exactly once.
* ``ack()``/``nack()`` are purely local bookkeeping.  A local ack is NOT a
  server-side acknowledgement; Redis Pub/Sub has no such concept and this
  adapter does not pretend otherwise.
* ``close(wait=True)`` stops the reader and waits for in-flight leases to
  be settled; ``close(wait=False)`` terminates immediately and returns a
  snapshot of all unacknowledged messages (leased-but-unsettled and
  buffered-but-never-leased).  After termination, late ``ack()``/``nack()``
  calls are no-ops returning ``False`` -- the snapshot is the authoritative
  record of what was left unacknowledged.
"""

import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

__all__ = [
    "BoundedPubSubBuffer",
    "ConsumerClosedError",
    "PubSubLease",
]


class ConsumerClosedError(RuntimeError):
    """Raised when leasing a message from a closed/closing buffer."""


class PubSubLease:
    """
    A handle on a single buffered message.

    The lease is settled by exactly one successful ``ack()`` or ``nack()``;
    both return ``True`` for the call that settled the lease and ``False``
    for any subsequent (duplicate) call.
    """

    __slots__ = ("_buffer", "message", "redeliveries", "_settled")

    def __init__(self, buffer: "BoundedPubSubBuffer", message: Any):
        self._buffer = buffer
        self.message = message
        self.redeliveries = 0
        self._settled = False

    @property
    def settled(self) -> bool:
        return self._settled

    def ack(self) -> bool:
        """
        Locally acknowledge the message, releasing its buffer slot.

        Returns True if this call settled the lease, False if the lease was
        already settled or the buffer has been terminated.  This is a local
        operation only; it is not a server-side acknowledgement.
        """
        buf = self._buffer
        with buf._cond:
            if self._settled or buf._terminated:
                return False
            self._settled = True
            buf._inflight.pop(id(self), None)
            buf._cond.notify_all()
            return True

    def nack(self) -> bool:
        """
        Locally negative-acknowledge the message.

        The message is returned to the front of the local queue and its
        redelivery counter is incremented.  When the counter reaches
        ``max_redeliveries`` the message is dropped and the buffer's
        ``on_failure`` callback is invoked exactly once.

        Returns True if this call settled the lease, False if the lease was
        already settled or the buffer has been terminated.
        """
        buf = self._buffer
        failure = None
        with buf._cond:
            if self._settled or buf._terminated:
                return False
            self._settled = True
            buf._inflight.pop(id(self), None)
            self.redeliveries += 1
            if self.redeliveries >= buf._max_redeliveries:
                failure = (self.message, self.redeliveries)
            else:
                # Back to the front of the queue for redelivery.
                self._settled = False
                buf._ready.appendleft(self)
            buf._cond.notify_all()
        if failure is not None and buf._on_failure is not None:
            buf._on_failure(failure[0], failure[1])
        return True


class BoundedPubSubBuffer:
    """
    Bounded local buffer with explicit ack/nack on top of a PubSub source.

    :param source: object with ``get_message(timeout=...)`` returning a
        message or ``None`` (e.g. ``redis.client.PubSub``).
    :param capacity: maximum number of messages held locally (buffered plus
        in-flight).  When reached, the reader stops pulling from the source.
    :param max_redeliveries: number of nacks tolerated per message before
        it is sent to ``on_failure`` (must be >= 1).
    :param on_failure: callable ``(message, redeliveries)`` invoked exactly
        once per message that exhausts its redeliveries.
    :param poll_timeout: timeout passed to the source's ``get_message``.
    """

    def __init__(
        self,
        source: Any,
        capacity: int,
        max_redeliveries: int = 3,
        on_failure: Optional[Callable[[Any, int], None]] = None,
        poll_timeout: float = 0.05,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if max_redeliveries < 1:
            raise ValueError("max_redeliveries must be >= 1")
        self._source = source
        self._capacity = capacity
        self._max_redeliveries = max_redeliveries
        self._on_failure = on_failure
        self._poll_timeout = poll_timeout

        self._cond = threading.Condition()
        self._ready: Deque[PubSubLease] = deque()
        self._inflight: Dict[int, PubSubLease] = {}
        self._closing = False
        self._terminated = False

        self._reader = threading.Thread(
            target=self._reader_loop,
            name="pubsub-buffer-reader",
            daemon=True,
        )
        self._reader.start()

    def get(self, timeout: Optional[float] = None) -> Optional[PubSubLease]:
        """
        Lease the next buffered message.

        Returns a :class:`PubSubLease`, or ``None`` on timeout.  Raises
        :class:`ConsumerClosedError` if the buffer is closing/closed and no
        buffered message remains.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if self._ready:
                    lease = self._ready.popleft()
                    lease._settled = False
                    self._inflight[id(lease)] = lease
                    return lease
                if self._closing:
                    raise ConsumerClosedError("the buffer is closed")
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
                else:
                    self._cond.wait()

    def close(self, wait: bool = True, timeout: Optional[float] = None) -> List[Any]:
        """
        Close the buffer and return a snapshot of unacknowledged messages.

        With ``wait=True`` the reader is stopped and this blocks (up to
        ``timeout`` seconds, indefinitely if None) until all in-flight
        leases are settled via ack/nack.  With ``wait=False`` it terminates
        immediately.

        Either way, returns the messages still unacknowledged at
        termination: leased-but-unsettled first, then buffered ones in
        queue order.  The snapshot is never silently dropped; after close,
        late ack/nack calls are no-ops returning False.
        """
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self._reader.join()

        if wait:
            deadline = None if timeout is None else time.monotonic() + timeout
            with self._cond:
                while self._inflight:
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._cond.wait(remaining)
                    else:
                        self._cond.wait()

        with self._cond:
            self._terminated = True
            snapshot = [lease.message for lease in self._inflight.values()]
            snapshot.extend(lease.message for lease in self._ready)
            self._inflight.clear()
            self._ready.clear()
            self._cond.notify_all()
            return snapshot

    def _occupancy(self) -> int:
        return len(self._ready) + len(self._inflight)

    def _reader_loop(self) -> None:
        while True:
            with self._cond:
                while not self._closing and self._occupancy() >= self._capacity:
                    self._cond.wait()
                if self._closing:
                    return
            try:
                message = self._source.get_message(timeout=self._poll_timeout)
            except Exception:
                # Source error: back off briefly instead of spinning.
                with self._cond:
                    if self._closing:
                        return
                time.sleep(self._poll_timeout)
                continue
            if message is None:
                continue
            with self._cond:
                # Even if closing started while we were blocked in
                # get_message, buffer the message so it shows up in the
                # close snapshot instead of being lost.
                lease = PubSubLease(self, message)
                self._ready.append(lease)
                self._cond.notify_all()
                if self._closing:
                    return
