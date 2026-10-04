"""
Tests for redis.pubsub_buffer.BoundedPubSubBuffer.

All tests are local-only: they use a fake PubSub source and never touch a
Redis server.
"""

import threading
import time
from collections import deque

import pytest
from redis.pubsub_buffer import (
    BoundedPubSubBuffer,
    ConsumerClosedError,
)

M1 = {"type": "message", "channel": b"ch", "data": b"m1"}
M2 = {"type": "message", "channel": b"ch", "data": b"m2"}


class FakePubSub:
    """Minimal stand-in for redis.client.PubSub (get_message only)."""

    def __init__(self, messages):
        self._messages = deque(messages)
        self._lock = threading.Lock()
        self.calls = []
        self.taken = 0

    def get_message(self, timeout=None):
        with self._lock:
            self.calls.append("get_message")
            if self._messages:
                self.taken += 1
                return self._messages.popleft()
            return None


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def buffers():
    created = []

    def make(*args, **kwargs):
        buf = BoundedPubSubBuffer(*args, **kwargs)
        created.append(buf)
        return buf

    yield make
    for buf in created:
        buf.close(wait=False)


class TestAcceptance:
    def test_nack_redelivers_to_front_before_buffered_message(self, buffers):
        # capacity 2, fake input m1, m2: after leasing m1 and nacking it,
        # m1 must be redelivered before m2.
        buf = buffers(FakePubSub([M1, M2]), capacity=2)

        lease = buf.get(timeout=1)
        assert lease.message is M1
        assert lease.nack() is True

        redelivered = buf.get(timeout=1)
        assert redelivered.message is M1
        assert redelivered.redeliveries == 1
        assert redelivered.ack() is True

        second = buf.get(timeout=1)
        assert second.message is M2
        assert second.ack() is True

    def test_three_nacks_reach_limit_and_fire_failure_callback_once(self, buffers):
        failures = []
        buf = buffers(
            FakePubSub([M1]),
            capacity=1,
            max_redeliveries=3,
            on_failure=lambda msg, n: failures.append((msg, n)),
        )

        for expected_redeliveries in (1, 2):
            lease = buf.get(timeout=1)
            assert lease.message is M1
            assert lease.nack() is True
            assert lease.redeliveries == expected_redeliveries
            assert failures == []

        lease = buf.get(timeout=1)
        assert lease.message is M1
        assert lease.nack() is True

        # Exactly one failure callback at the limit, then the message is gone.
        assert failures == [(M1, 3)]
        assert buf.get(timeout=0.2) is None


class TestAckSemantics:
    def test_duplicate_ack_is_a_noop_and_frees_slot_exactly_once(self, buffers):
        source = FakePubSub([M1, M2])
        buf = buffers(source, capacity=1)

        lease = buf.get(timeout=1)
        assert lease.ack() is True
        # Duplicate ack / nack after settle must not settle again.
        assert lease.ack() is False
        assert lease.nack() is False

        # The single ack freed exactly one slot: m2 can now be buffered.
        assert wait_for(lambda: source.taken == 2)
        second = buf.get(timeout=1)
        assert second.message is M2
        assert second.ack() is True

    def test_unacked_lease_is_never_redelivered(self, buffers):
        buf = buffers(FakePubSub([M1]), capacity=1)
        lease = buf.get(timeout=1)
        # While the lease is unsettled, nothing else is handed out.
        assert buf.get(timeout=0.2) is None
        assert lease.ack() is True

    def test_ack_is_local_only_and_never_touches_the_source(self, buffers):
        source = FakePubSub([M1])
        buf = buffers(source, capacity=1)
        lease = buf.get(timeout=1)
        assert lease.ack() is True
        buf.close(wait=False)
        # The only interaction with the underlying PubSub is reading;
        # ack/nack/close issue no server-side calls.
        assert set(source.calls) == {"get_message"}


class TestBackpressure:
    def test_reader_stops_when_buffer_is_full(self, buffers):
        source = FakePubSub([M1, M2])
        buf = buffers(source, capacity=1)

        lease = buf.get(timeout=1)
        assert lease.message is M1
        # occupancy == capacity (1 in-flight): the reader must not pull m2.
        time.sleep(0.2)
        assert source.taken == 1

        # After ack the slot frees up and the reader resumes.
        assert lease.ack() is True
        assert wait_for(lambda: source.taken == 2)
        second = buf.get(timeout=1)
        assert second.message is M2
        assert second.ack() is True

    def test_nacked_message_still_counts_against_capacity(self, buffers):
        source = FakePubSub([M1, M2])
        buf = buffers(source, capacity=1)

        lease = buf.get(timeout=1)
        assert lease.nack() is True
        # m1 is back in the local queue, so the buffer is still full.
        time.sleep(0.2)
        assert source.taken == 1

        redelivered = buf.get(timeout=1)
        assert redelivered.message is M1
        assert redelivered.ack() is True
        assert wait_for(lambda: source.taken == 2)


class TestClose:
    def test_immediate_close_returns_unacked_snapshot(self, buffers):
        source = FakePubSub([M1, M2])
        buf = buffers(source, capacity=2)
        lease = buf.get(timeout=1)
        assert wait_for(lambda: len(buf._ready) == 1)

        snapshot = buf.close(wait=False)
        # leased-but-unacked first, then buffered, in queue order.
        assert snapshot == [M1, M2]
        # Late ack is a no-op; the snapshot already recorded the truth.
        assert lease.ack() is False
        assert lease.nack() is False

    def test_graceful_close_waits_for_inflight_ack(self, buffers):
        buf = buffers(FakePubSub([M1]), capacity=1)
        lease = buf.get(timeout=1)

        result = {}

        def do_close():
            result["snapshot"] = buf.close(wait=True)

        t = threading.Thread(target=do_close)
        t.start()
        time.sleep(0.2)
        assert t.is_alive()  # blocked waiting for the in-flight lease

        assert lease.ack() is True
        t.join(2)
        assert not t.is_alive()
        assert result["snapshot"] == []

    def test_graceful_close_with_timeout_returns_remaining_unacked(self, buffers):
        buf = buffers(FakePubSub([M1]), capacity=1)
        lease = buf.get(timeout=1)  # noqa: F841 - intentionally never acked
        snapshot = buf.close(wait=True, timeout=0.2)
        assert snapshot == [M1]

    def test_get_after_close_raises(self, buffers):
        buf = buffers(FakePubSub([]), capacity=1)
        buf.close(wait=False)
        with pytest.raises(ConsumerClosedError):
            buf.get(timeout=1)

    def test_close_and_ack_interleaved_with_barrier_never_loses_unacked(self, buffers):
        # Hammer the close/ack race: whichever wins, m1 is accounted for
        # exactly once -- either by the ack or by the close snapshot.
        for _ in range(50):
            buf = buffers(FakePubSub([M1]), capacity=2)
            lease = buf.get(timeout=1)
            barrier = threading.Barrier(2)
            results = {}

            def do_close():
                barrier.wait()
                results["snapshot"] = buf.close(wait=False)

            def do_ack():
                barrier.wait()
                results["acked"] = lease.ack()

            threads = [
                threading.Thread(target=do_close),
                threading.Thread(target=do_ack),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(2)

            if results["acked"]:
                assert results["snapshot"] == []
            else:
                assert results["snapshot"] == [M1]
