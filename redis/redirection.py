"""
Per-request cluster redirection tracing (MOVED / ASK).

The cluster client retries a command across nodes when it receives a
``-MOVED`` or ``-ASK`` redirection. Normally those hops are invisible to the
caller; this module provides :class:`RedirectionTracker`, an opt-in object the
caller can pass (via ``redirect_tracker=`` to ``execute_command``) to collect
a structured report of where the request went.

Design notes:

- Only routing data is ever recorded: the redirection kind, the slot id, and
  the ``host:port`` node names reported by the server. Command arguments,
  credentials and connection parameters are deliberately never read, so a
  report cannot contain authentication information.
- MOVED and ASK are kept distinct (:class:`RedirectionType`). An ASK
  redirection is a transient hint for a single request and does not update the
  slots cache; a MOVED redirection is a permanent topology change. Mixing the
  two in one bucket would make a report misleading.
- A MOVED edge the request has already followed is flagged as a redirect loop
  (``-MOVED`` cycles eventually exhaust the client's request TTL). ASK hops are
  not part of loop detection: an ASK never changes routing, so it cannot form
  a routing cycle.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "RedirectionType",
    "RedirectionHop",
    "RedirectionReport",
    "RedirectionTracker",
]


class RedirectionType(str, Enum):
    """Kind of cluster redirection, kept distinct on purpose."""

    MOVED = "MOVED"
    ASK = "ASK"


@dataclass(frozen=True)
class RedirectionHop:
    """
    A single redirection followed while serving one request.

    :ivar type: whether the hop was a ``MOVED`` or an ``ASK`` redirection.
    :ivar slot_id: the hash slot the server reported.
    :ivar from_node: ``host:port`` of the node that returned the redirection.
    :ivar to_node: ``host:port`` of the node the client was told to contact.
    :ivar redirect_loop: ``True`` when this MOVED edge had already been
        followed earlier in the same request (a circular redirection).
    """

    type: RedirectionType
    slot_id: int
    from_node: str
    to_node: str
    redirect_loop: bool = False

    def __str__(self) -> str:
        loop = " (loop)" if self.redirect_loop else ""
        return (
            f"{self.type.value} slot {self.slot_id}: "
            f"{self.from_node} -> {self.to_node}{loop}"
        )


@dataclass(frozen=True)
class RedirectionReport:
    """
    The collected redirection trace for a completed request.

    :ivar hops: every MOVED/ASK redirection followed, in order.
    :ivar moved_count: number of MOVED redirections.
    :ivar ask_count: number of ASK redirections.
    :ivar redirect_loop_detected: whether a circular MOVED edge was seen.
    :ivar final_node: ``host:port`` of the node that produced the final
        response, or ``None`` if the request never completed.
    """

    hops: Tuple[RedirectionHop, ...] = ()
    moved_count: int = 0
    ask_count: int = 0
    redirect_loop_detected: bool = False
    final_node: Optional[str] = None

    @property
    def total_redirections(self) -> int:
        return len(self.hops)

    def summary(self) -> str:
        lines = [
            f"redirections={self.total_redirections} "
            f"(moved={self.moved_count}, ask={self.ask_count})",
            f"redirect_loop_detected={self.redirect_loop_detected}",
            f"final_node={self.final_node}",
        ]
        lines.extend(f"  {index + 1}. {hop}" for index, hop in enumerate(self.hops))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class RedirectionTracker:
    """
    Collects the redirection hops of a single cluster request.

    Pass an instance as ``redirect_tracker=`` to
    ``RedisCluster.execute_command`` and inspect :attr:`report` once the call
    returns::

        tracker = RedirectionTracker()
        await cluster.execute_command("GET", "foo", redirect_tracker=tracker)
        print(tracker.report)

    The tracker holds only routing data (redirection kind, slot id, node
    ``host:port`` names); it never reads command arguments or connection
    parameters, so no password, username, token or other authentication
    information can end up in the report.
    """

    __slots__ = (
        "_hops",
        "_seen_moved_edges",
        "_moved_count",
        "_ask_count",
        "_loop_detected",
        "_final_node",
    )

    def __init__(self) -> None:
        self._hops: List[RedirectionHop] = []
        # Maps slot id to the directed MOVED edges already followed.
        self._seen_moved_edges: Dict[int, List[Tuple[str, str]]] = {}
        self._moved_count = 0
        self._ask_count = 0
        self._loop_detected = False
        self._final_node: Optional[str] = None

    def record_moved(
        self, slot_id: int, from_node: str, to_node: str
    ) -> RedirectionHop:
        """Record a ``-MOVED`` redirection and return the stored hop."""
        edges = self._seen_moved_edges.setdefault(slot_id, [])
        edge = (from_node, to_node)
        is_loop = edge in edges
        hop = RedirectionHop(
            type=RedirectionType.MOVED,
            slot_id=slot_id,
            from_node=from_node,
            to_node=to_node,
            redirect_loop=is_loop,
        )
        self._hops.append(hop)
        edges.append(edge)
        self._moved_count += 1
        if is_loop:
            self._loop_detected = True
        return hop

    def record_ask(self, slot_id: int, from_node: str, to_node: str) -> RedirectionHop:
        """
        Record an ``-ASK`` redirection and return the stored hop.

        ASK is a transient one-off hint: it is never counted as a routing
        change and never participates in redirect-loop detection.
        """
        hop = RedirectionHop(
            type=RedirectionType.ASK,
            slot_id=slot_id,
            from_node=from_node,
            to_node=to_node,
        )
        self._hops.append(hop)
        self._ask_count += 1
        return hop

    def set_final_node(self, node_name: str) -> None:
        """Record the node that produced the request's final response."""
        self._final_node = node_name

    @property
    def redirect_loop_detected(self) -> bool:
        return self._loop_detected

    @property
    def hops(self) -> Tuple[RedirectionHop, ...]:
        return tuple(self._hops)

    @property
    def report(self) -> RedirectionReport:
        return RedirectionReport(
            hops=tuple(self._hops),
            moved_count=self._moved_count,
            ask_count=self._ask_count,
            redirect_loop_detected=self._loop_detected,
            final_node=self._final_node,
        )

    def reset(self) -> None:
        """Clear all recorded state, so the tracker can be reused."""
        self._hops.clear()
        self._seen_moved_edges.clear()
        self._moved_count = 0
        self._ask_count = 0
        self._loop_detected = False
        self._final_node = None
