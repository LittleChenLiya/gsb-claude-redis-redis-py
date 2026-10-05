"""Explicit Redis Cluster topology snapshot compiler.

This module compiles an *explicit* cluster topology -- a set of nodes and
the slot ranges they own -- into an immutable attribution index. It validates
that every hash slot in 0..16383 is covered exactly once (no gaps, no
overlaps), every slot range references a known node, only primary nodes own
slots (replicas never do), and node identities are kept separate from network
addresses.
"""

from redis.cluster import PRIMARY, REPLICA
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot

__all__ = [
    "PRIMARY",
    "REPLICA",
    "SlotRange",
    "TopologyCompileError",
    "TopologyNode",
    "TopologySnapshot",
    "compile_topology",
]


class TopologyCompileError(Exception):
    """Raised when an explicit topology snapshot cannot be compiled."""


class TopologyNode:
    """A node in an explicit cluster topology.

    A node's identity is its node_id, kept separate from its network
    address (host/port). Two nodes may share an address but have distinct
    node IDs; the node ID is the authoritative identity used when resolving
    slot ownership.
    """

    def __init__(self, node_id, host, port, role=PRIMARY):
        if not isinstance(node_id, str) or not node_id:
            raise TopologyCompileError(
                f"node_id must be a non-empty string, got {node_id!r}"
            )
        if role not in (PRIMARY, REPLICA):
            raise TopologyCompileError(
                f"node {node_id!r}: role must be PRIMARY or REPLICA, got {role!r}"
            )
        self.node_id = node_id
        self.host = host
        self.port = port
        self.role = role

    @property
    def address(self):
        """The network address (host, port)."""
        return (self.host, self.port)

    def __repr__(self):
        return (
            f"TopologyNode(node_id={self.node_id!r}, "
            f"address={self.host}:{self.port}, role={self.role!r})"
        )


class SlotRange:
    """An inclusive range of hash slots owned by a node.

    source records provenance (e.g. the shard or config section this range
    came from) and is reported back by the snapshot.
    """

    def __init__(self, start, end, node_id, source=None):
        if not isinstance(start, int) or not isinstance(end, int):
            raise TopologyCompileError(
                f"slot range endpoints must be integers, got start={start!r}, end={end!r}"
            )
        if start < 0 or end >= REDIS_CLUSTER_HASH_SLOTS:
            raise TopologyCompileError(
                f"slot range {start}-{end} is outside 0..{REDIS_CLUSTER_HASH_SLOTS - 1}"
            )
        if start > end:
            raise TopologyCompileError(
                f"slot range start {start} is greater than end {end}"
            )
        if not isinstance(node_id, str) or not node_id:
            raise TopologyCompileError(
                f"slot range {start}-{end}: node_id must be a non-empty string, got {node_id!r}"
            )
        self.start = start
        self.end = end
        self.node_id = node_id
        self.source = source

    def slots(self):
        """The slots covered by this range."""
        return range(self.start, self.end + 1)

    def __repr__(self):
        return (
            f"SlotRange({self.start}-{self.end}, node_id={self.node_id!r}, "
            f"source={self.source!r})"
        )


class TopologySnapshot:
    """An immutable, compiled cluster topology snapshot.

    Maps every hash slot to the primary node that owns it. Only produced by
    compile_topology after every validation passes, so ownership lookups are
    total and consistent. The snapshot is immutable: internal mappings are
    stored as tuples and cannot be modified after construction.
    """

    def __init__(self, nodes, slot_owners, range_sources):
        # nodes: dict node_id -> TopologyNode
        # slot_owners: tuple of node_id per slot (length REDIS_CLUSTER_HASH_SLOTS)
        # range_sources: tuple of (start, end, node_id, source)
        self._nodes = dict(nodes)
        self._slot_owners = tuple(slot_owners)
        self._range_sources = tuple(range_sources)

    @property
    def slot_count(self):
        """The number of slots in the snapshot (always 16384)."""
        return len(self._slot_owners)

    def owner_node_id(self, slot):
        """Return the node ID of the primary owning slot."""
        if not isinstance(slot, int) or not 0 <= slot < REDIS_CLUSTER_HASH_SLOTS:
            raise TopologyCompileError(
                f"slot must be in 0..{REDIS_CLUSTER_HASH_SLOTS - 1}, got {slot!r}"
            )
        return self._slot_owners[slot]

    def owner(self, slot):
        """Return the primary TopologyNode owning slot."""
        return self._nodes[self.owner_node_id(slot)]

    def node_for_key(self, key):
        """Return the primary node owning the slot of key.

        Uses redis.crc.key_slot (existing CRC16/XMODEM with hash-tag handling);
        no separate CRC is implemented here.
        """
        return self.owner(key_slot(key))

    def slot_source(self, slot):
        """Return the provenance label of the range covering slot.

        Returns None if the range had no source label.
        """
        node_id = self.owner_node_id(slot)
        for start, end, owner_id, source in self._range_sources:
            if start <= slot <= end and owner_id == node_id:
                return source
        return None

    def sources(self):
        """Yield (start, end, node_id, source) for every input range."""
        yield from self._range_sources

    def nodes(self):
        """Return all nodes in the snapshot, keyed by node_id."""
        return dict(self._nodes)

    def primaries(self):
        """Return all primary nodes in the snapshot."""
        return {
            node_id: node
            for node_id, node in self._nodes.items()
            if node.role == PRIMARY
        }

    def replicas(self):
        """Return all replica nodes in the snapshot."""
        return {
            node_id: node
            for node_id, node in self._nodes.items()
            if node.role == REPLICA
        }

    def __repr__(self):
        return (
            f"TopologySnapshot(nodes={len(self._nodes)}, "
            f"slots={self.slot_count}, ranges={len(self._range_sources)})"
        )


def compile_topology(nodes, slot_ranges):
    """Compile an explicit topology snapshot from nodes and slot ranges.

    Validates that every node has a unique non-empty node_id and a valid
    role, every slot range is within 0..16383 and references a known node,
    only primary nodes own slots (replicas never do), and every slot is
    covered exactly once (no gaps, no overlaps).

    Returns an immutable TopologySnapshot on success.
    Raises TopologyCompileError on any validation failure.

    This is a pure function: it never connects to Redis, never queries a
    live cluster, and never tries to auto-fix unknown nodes.
    """
    # --- validate nodes ---
    nodes_by_id = {}
    for node in nodes:
        if not isinstance(node, TopologyNode):
            raise TopologyCompileError(
                f"expected TopologyNode, got {type(node).__name__}"
            )
        if node.node_id in nodes_by_id:
            raise TopologyCompileError(
                f"duplicate node_id {node.node_id!r}: node IDs must be unique"
            )
        nodes_by_id[node.node_id] = node

    # --- validate slot ranges and build the slot -> owner map ---
    slot_owners = [None] * REDIS_CLUSTER_HASH_SLOTS
    range_sources = []
    for slot_range in slot_ranges:
        if not isinstance(slot_range, SlotRange):
            raise TopologyCompileError(
                f"expected SlotRange, got {type(slot_range).__name__}"
            )
        owner_id = slot_range.node_id
        source = slot_range.source

        # the owner must be a known node; unknown nodes cannot be auto-fixed
        owner = nodes_by_id.get(owner_id)
        if owner is None:
            raise TopologyCompileError(
                f"slot range {slot_range.start}-{slot_range.end} "
                f"(source={source!r}) references unknown node_id {owner_id!r}; "
                f"unknown nodes cannot be auto-fixed"
            )

        # only primaries own slots; replicas never do
        if owner.role != PRIMARY:
            raise TopologyCompileError(
                f"slot range {slot_range.start}-{slot_range.end} "
                f"(source={source!r}) is owned by {owner_id!r} which is a "
                f"{owner.role}; replicas do not own slots directly"
            )

        # record provenance
        range_sources.append((slot_range.start, slot_range.end, owner_id, source))

        # fill the slot map, detecting overlaps
        for slot in slot_range.slots():
            if slot_owners[slot] is not None:
                prev_source = _find_source(range_sources, slot)
                raise TopologyCompileError(
                    f"slot {slot} is covered by more than one range: "
                    f"source={prev_source!r} and source={source!r}"
                )
            slot_owners[slot] = owner_id

    # --- check for gaps (full coverage) ---
    missing = [slot for slot, owner in enumerate(slot_owners) if owner is None]
    if missing:
        gap_summary = _summarize_gaps(missing)
        raise TopologyCompileError(
            f"topology is not fully covered: {len(missing)} slot(s) missing: "
            f"{gap_summary}"
        )

    return TopologySnapshot(nodes_by_id, slot_owners, range_sources)


def _find_source(range_sources, slot):
    for start, end, _owner_id, source in range_sources:
        if start <= slot <= end:
            return source
    return None


def _summarize_gaps(missing):
    """Summarize a sorted list of missing slots as compact ranges."""
    if not missing:
        return "[]"
    ranges = []
    start = prev = missing[0]
    for slot in missing[1:]:
        if slot == prev + 1:
            prev = slot
        else:
            ranges.append((start, prev))
            start = prev = slot
    ranges.append((start, prev))
    parts = []
    for start, end in ranges[:5]:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    if len(ranges) > 5:
        parts.append(f"... and {len(ranges) - 5} more")
    return "[" + ", ".join(parts) + "]"
