"""Explicit Redis Cluster topology snapshot compiler.

This module turns a *declared* cluster topology (a set of nodes plus the
slot ranges they claim to own) into an immutable slot-ownership index.
It performs no I/O whatsoever: it never connects to Redis, never asks a
live node for its view of the cluster and never "repairs" an input that
mentions an unknown node. A snapshot is produced only when every
invariant below holds.

Invariants checked
------------------
* Slots ``0 .. slot_count-1`` (by default the Redis Cluster 16384) are
  covered exactly once.
* No two ranges overlap and no slot is missing; two ranges that merely
  share an edge (``[0, 8192]`` + ``[8192, 16383]``) are an overlap and
  are rejected.
* Every slot range is owned by a node that exists and is a ``master``.
  Replicas never own slots directly.
* Master/replica roles are consistent: a replica names exactly one
  existing master as its replication source, and a master names none.

Node identity (``node_id``) is deliberately kept separate from network
address (``host``/``port``): ownership is keyed by node id, never by
address, so moving a node to a new address does not silently change slot
ownership. Every accepted slot range carries its own ``source`` string
so the compiled snapshot can report where each range came from.

Key to slot resolution reuses :func:`redis.crc.key_slot`, i.e. the same
CRC16/XMODEM + hash-tag implementation the cluster client uses; this
module does not ship a second CRC implementation.

Run the local, server-free tests with::

    python -m pytest tests/test_topology.py -q
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot

ROLE_MASTER = "master"
ROLE_REPLICA = "replica"
_ROLES = (ROLE_MASTER, ROLE_REPLICA)

__all__ = [
    "ROLE_MASTER",
    "ROLE_REPLICA",
    "Node",
    "SlotRange",
    "TopologySnapshot",
    "TopologyCompileError",
    "compile_topology",
]


class TopologyCompileError(ValueError):
    """Raised when a declared topology cannot be compiled.

    All problems found in one compilation pass are collected in
    :attr:`errors` rather than reported one at a time.
    """

    def __init__(self, errors: Sequence[str]):
        self.errors: Tuple[str, ...] = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class Node:
    """A declared cluster node.

    Identity (``node_id``) and network location (``host``/``port``) are
    separate on purpose: ownership indexes use the id, and the address is
    only kept for the report. ``master_id`` is ``None`` for masters and
    must point at an existing master for replicas.
    """

    node_id: str
    host: str
    port: int
    role: str = ROLE_MASTER
    master_id: Optional[str] = None

    @property
    def address(self) -> Tuple[str, int]:
        return self.host, self.port


@dataclass(frozen=True)
class SlotRange:
    """An inclusive ``[start, end]`` slot claim made by one master.

    ``source`` is free-form provenance (file name + line, config shard
    name, ...) preserved verbatim on the compiled snapshot so every slot
    range can be traced back to where it was declared.
    """

    start: int
    end: int
    owner_id: str
    source: str = ""

    def __contains__(self, slot: object) -> bool:
        if not isinstance(slot, int):
            return False
        return self.start <= slot <= self.end


@dataclass(frozen=True)
class TopologySnapshot:
    """The immutable result of a successful compilation.

    The index tuples are fixed length ``slot_count`` and map every slot
    to its owner node id and to the :class:`SlotRange` that declared it.
    """

    slot_count: int
    nodes: Mapping[str, Node]
    addresses: Mapping[Tuple[str, int], str]
    ranges: Tuple[SlotRange, ...]
    slot_owners: Tuple[str, ...]
    slot_ranges: Tuple[Optional[SlotRange], ...]

    def owner_of_slot(self, slot: int) -> Node:
        """Return the master node that owns ``slot``."""
        self._check_slot(slot)
        return self.nodes[self.slot_owners[slot]]

    def owner_for_key(self, key: bytes) -> Node:
        """Return the master that owns ``key``'s slot.

        Slot computation (including ``{hash tag}``) is delegated to
        :func:`redis.crc.key_slot`; only byte keys are accepted.
        """
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes; encode str keys before lookup")
        return self.owner_of_slot(key_slot(key, self.slot_count))

    def range_of_slot(self, slot: int) -> SlotRange:
        """Return the declared :class:`SlotRange` covering ``slot``."""
        self._check_slot(slot)
        return self.slot_ranges[slot]  # type: ignore[return-value]

    def range_source(self, slot: int) -> str:
        """Report the provenance string of the range covering ``slot``."""
        return self.range_of_slot(slot).source

    def range_report(self) -> Tuple[Tuple[int, int, str, str], ...]:
        """Report ``(start, end, owner_id, source)`` for each range.

        Ranges are reported in slot order.
        """
        return tuple(
            (rng.start, rng.end, rng.owner_id, rng.source) for rng in self.ranges
        )

    def replicas_of(self, master_id: str) -> Tuple[Node, ...]:
        """Return the declared replicas of ``master_id``."""
        if master_id not in self.nodes:
            raise KeyError(master_id)
        return tuple(
            node
            for node in self.nodes.values()
            if node.role == ROLE_REPLICA and node.master_id == master_id
        )

    def _check_slot(self, slot: int) -> None:
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise TypeError("slot must be an int")
        if not 0 <= slot < self.slot_count:
            raise ValueError(f"slot {slot} out of range 0..{self.slot_count - 1}")


def compile_topology(
    nodes: Sequence[Node],
    ranges: Sequence[SlotRange],
    slot_count: int = REDIS_CLUSTER_HASH_SLOTS,
) -> TopologySnapshot:
    """Compile declared nodes and slot ranges into an immutable snapshot.

    Pure function: opens no sockets, performs no discovery and never
    mutates or "fixes" the inputs. Raises :class:`TopologyCompileError`
    containing every problem found if the topology is not exactly
    consistent.
    """
    errors = []

    if (
        not isinstance(slot_count, int)
        or isinstance(slot_count, bool)
        or slot_count <= 0
    ):
        raise TopologyCompileError(
            [f"slot_count must be a positive int, got {slot_count!r}"]
        )

    nodes = _as_list(nodes, "nodes", Node, errors)
    ranges = _as_list(ranges, "ranges", SlotRange, errors)
    if errors:
        raise TopologyCompileError(errors)

    by_id, by_addr = _compile_nodes(nodes, errors)

    # Cross-node role checks need the full id table, hence a second pass.
    for node in nodes:
        if node.role == ROLE_REPLICA:
            if not node.master_id:
                errors.append(f"node {node.node_id!r} is a replica with no master_id")
            elif node.master_id not in by_id:
                errors.append(
                    f"node {node.node_id!r} replicates unknown node {node.master_id!r}"
                )
            elif by_id[node.master_id].role != ROLE_MASTER:
                errors.append(
                    f"node {node.node_id!r} replicates {node.master_id!r}, "
                    "which is a replica, not a master"
                )
        elif node.master_id is not None and node.master_id != node.node_id:
            errors.append(
                f"master node {node.node_id!r} must not name a different "
                f"master_id ({node.master_id!r})"
            )

    valid_ranges = _validate_ranges(ranges, slot_count, by_id, errors)

    ordered = sorted(valid_ranges, key=lambda rng: (rng.start, rng.end))
    _check_coverage(ordered, slot_count, errors)
    if errors:
        raise TopologyCompileError(errors)

    slot_owners = [""] * slot_count
    slot_ranges: list[Optional[SlotRange]] = [None] * slot_count
    for rng in ordered:
        for slot in range(rng.start, rng.end + 1):
            slot_owners[slot] = rng.owner_id
            slot_ranges[slot] = rng

    return TopologySnapshot(
        slot_count=slot_count,
        nodes=MappingProxyType(by_id),
        addresses=MappingProxyType(dict(by_addr)),
        ranges=tuple(ordered),
        slot_owners=tuple(slot_owners),
        slot_ranges=tuple(slot_ranges),
    )


def _as_list(value: object, name: str, item_type: type, errors: list) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        errors.append(f"{name} must be a sequence of {item_type.__name__}")
        return []
    items = list(value)
    for index, item in enumerate(items):
        if not isinstance(item, item_type):
            errors.append(
                f"{name}[{index}] must be {item_type.__name__}, "
                f"got {type(item).__name__}"
            )
    return items


def _compile_nodes(
    nodes: Sequence[Node], errors: list
) -> Tuple[dict, MappingProxyType]:
    by_id: dict = {}
    by_addr: dict = {}
    for index, node in enumerate(nodes):
        where = f"nodes[{index}]"
        if not isinstance(node.node_id, str) or not node.node_id:
            errors.append(f"{where}: node_id must be a non-empty string")
        elif node.node_id in by_id:
            errors.append(f"{where}: duplicate node_id {node.node_id!r}")
        if not isinstance(node.host, str) or not node.host:
            errors.append(
                f"{where} ({node.node_id!r}): host must be a non-empty string"
            )
        if (
            not isinstance(node.port, int)
            or isinstance(node.port, bool)
            or not 1 <= node.port <= 65535
        ):
            errors.append(
                f"{where} ({node.node_id!r}): port must be an int in 1..65535"
            )
        elif node.host:
            address = (node.host, node.port)
            if address in by_addr:
                errors.append(
                    f"{where} ({node.node_id!r}): address {address[0]}:"
                    f"{address[1]} already used by node {by_addr[address]!r}"
                )
            else:
                by_addr[address] = node.node_id
        if node.role not in _ROLES:
            errors.append(
                f"{where} ({node.node_id!r}): role must be one of {_ROLES}, "
                f"got {node.role!r}"
            )
        if node.master_id is not None and not isinstance(node.master_id, str):
            errors.append(
                f"{where} ({node.node_id!r}): master_id must be a str or None"
            )
        if isinstance(node.node_id, str) and node.node_id:
            by_id.setdefault(node.node_id, node)
    return by_id, MappingProxyType(by_addr)


def _validate_ranges(
    ranges: Sequence[SlotRange],
    slot_count: int,
    by_id: Mapping[str, Node],
    errors: list,
) -> list:
    valid = []
    for index, rng in enumerate(ranges):
        where = f"ranges[{index}]"
        if rng.source:
            where += f" (source={rng.source!r})"
        bounds_ok = True
        for field_name in ("start", "end"):
            value = getattr(rng, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{where}: {field_name} must be an int")
                bounds_ok = False
        if bounds_ok:
            if not 0 <= rng.start < slot_count or not 0 <= rng.end < slot_count:
                errors.append(
                    f"{where}: bounds must lie within 0..{slot_count - 1}, "
                    f"got [{rng.start}, {rng.end}]"
                )
            elif rng.start > rng.end:
                errors.append(
                    f"{where}: start {rng.start} is greater than end {rng.end}"
                )
            else:
                valid.append(rng)
        owner = by_id.get(rng.owner_id)
        if not isinstance(rng.owner_id, str) or not rng.owner_id:
            errors.append(f"{where}: owner_id must be a non-empty string")
        elif owner is None:
            # Unknown nodes are never auto-repaired into the topology.
            errors.append(f"{where}: owner {rng.owner_id!r} is not declared in nodes")
        elif owner.role != ROLE_MASTER:
            # Replicas do not own slots directly; slots belong to their master.
            errors.append(
                f"{where}: owner {rng.owner_id!r} is a {owner.role}; "
                "slots can only be owned by a master"
            )
    return valid


def _check_coverage(
    ordered: Sequence[SlotRange], slot_count: int, errors: list
) -> None:
    if not ordered:
        errors.append(
            f"no slot ranges declared; slots 0..{slot_count - 1} are uncovered"
        )
        return
    first = ordered[0]
    if first.start != 0:
        errors.append(
            f"{_label(first)}: coverage starts at slot {first.start}, slot 0 is missing"
        )
    previous = first
    for rng in ordered[1:]:
        if rng.start <= previous.end:
            errors.append(
                f"{_label(rng)} overlaps {_label(previous)} on slot(s) "
                f"{rng.start}..{min(rng.end, previous.end)}"
            )
        elif rng.start != previous.end + 1:
            errors.append(
                f"gap between {_label(previous)} and {_label(rng)}: "
                f"slot(s) {previous.end + 1}..{rng.start - 1} uncovered"
            )
        if rng.end > previous.end:
            previous = rng
    last = ordered[-1]
    if last.end != slot_count - 1:
        errors.append(
            f"{_label(last)}: coverage ends at slot {last.end}, "
            f"slot(s) {last.end + 1}..{slot_count - 1} are missing"
        )


def _label(rng: SlotRange) -> str:
    text = f"[{rng.start}, {rng.end}] owned by {rng.owner_id}"
    if rng.source:
        text += f" (source={rng.source!r})"
    return text
