"""Tests for the explicit Redis Cluster topology snapshot compiler.

Run locally (no Redis server needed):

    pytest tests/test_cluster_topology.py -q

The compiler is a pure function of its inputs; it never connects to Redis.
The test_connect_count_is_zero test pins that guarantee.
"""

from unittest.mock import patch

import pytest

from redis.cluster import PRIMARY, REPLICA
from redis.cluster_topology import (
    SlotRange,
    TopologyCompileError,
    TopologyNode,
    compile_topology,
)
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot


def _two_node_topology():
    """A two-node cluster: node-1 owns the first half, node-2 the second."""
    nodes = [
        TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
        TopologyNode("node-2", "127.0.0.1", 7001, role=PRIMARY),
    ]
    ranges = [
        SlotRange(0, 8191, "node-1", source="shard-1"),
        SlotRange(8192, 16383, "node-2", source="shard-2"),
    ]
    return nodes, ranges


class TestTwoNodeExample:
    """The two-node example: each primary owns exactly one half."""

    def test_two_halves_pass_completely(self):
        nodes, ranges = _two_node_topology()
        snapshot = compile_topology(nodes, ranges)

        assert snapshot.slot_count == REDIS_CLUSTER_HASH_SLOTS
        owners = [
            snapshot.owner_node_id(slot) for slot in range(REDIS_CLUSTER_HASH_SLOTS)
        ]
        assert all(owner in ("node-1", "node-2") for owner in owners)
        assert owners[0] == "node-1"
        assert owners[8191] == "node-1"
        assert owners[8192] == "node-2"
        assert owners[16383] == "node-2"

    def test_owner_returns_correct_node(self):
        nodes, ranges = _two_node_topology()
        snapshot = compile_topology(nodes, ranges)

        assert snapshot.owner(0).node_id == "node-1"
        assert snapshot.owner(8191).node_id == "node-1"
        assert snapshot.owner(8192).node_id == "node-2"
        assert snapshot.owner(16383).node_id == "node-2"

    def test_node_for_key_routes_by_key_slot(self):
        nodes, ranges = _two_node_topology()
        snapshot = compile_topology(nodes, ranges)

        for key in (b"foo", b"bar", b"{team}a", b"{team}b", b"user:1000"):
            slot = key_slot(key)
            expected = snapshot.owner(slot).node_id
            assert snapshot.node_for_key(key).node_id == expected

    def test_hash_tag_maps_to_same_slot(self):
        nodes, ranges = _two_node_topology()
        snapshot = compile_topology(nodes, ranges)

        # {team}a and {team}b share a hash tag -> same slot -> same owner
        slot_a = key_slot(b"{team}a")
        slot_b = key_slot(b"{team}b")
        assert slot_a == slot_b
        assert snapshot.owner(slot_a).node_id == snapshot.owner(slot_b).node_id

    def test_sources_are_reported(self):
        nodes, ranges = _two_node_topology()
        snapshot = compile_topology(nodes, ranges)

        sources = list(snapshot.sources())
        assert sources == [
            (0, 8191, "node-1", "shard-1"),
            (8192, 16383, "node-2", "shard-2"),
        ]
        assert snapshot.slot_source(0) == "shard-1"
        assert snapshot.slot_source(8191) == "shard-1"
        assert snapshot.slot_source(8192) == "shard-2"
        assert snapshot.slot_source(16383) == "shard-2"

    def test_primaries_and_replicas(self):
        nodes, ranges = _two_node_topology()
        snapshot = compile_topology(nodes, ranges)

        assert set(snapshot.primaries()) == {"node-1", "node-2"}
        assert snapshot.replicas() == {}


class TestOverlapRejected:
    """edge overlap must be rejected."""

    def test_shared_edge_slot_rejected(self):
        # both ranges claim slot 8192 -> overlap
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
            TopologyNode("node-2", "127.0.0.1", 7001, role=PRIMARY),
        ]
        ranges = [
            SlotRange(0, 8192, "node-1", source="shard-1"),
            SlotRange(8192, 16383, "node-2", source="shard-2"),
        ]
        with pytest.raises(TopologyCompileError, match="more than one range"):
            compile_topology(nodes, ranges)

    def test_full_overlap_rejected(self):
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
            TopologyNode("node-2", "127.0.0.1", 7001, role=PRIMARY),
        ]
        ranges = [
            SlotRange(0, 16383, "node-1", source="shard-1"),
            SlotRange(0, 16383, "node-2", source="shard-2"),
        ]
        with pytest.raises(TopologyCompileError, match="more than one range"):
            compile_topology(nodes, ranges)


class TestMissingSlotRejected:
    """missing one slot must be rejected."""

    def test_missing_one_slot_rejected(self):
        # slot 8192 is missing (gap between 8191 and 8193)
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
            TopologyNode("node-2", "127.0.0.1", 7001, role=PRIMARY),
        ]
        ranges = [
            SlotRange(0, 8191, "node-1", source="shard-1"),
            SlotRange(8193, 16383, "node-2", source="shard-2"),
        ]
        with pytest.raises(TopologyCompileError, match="not fully covered"):
            compile_topology(nodes, ranges)

    def test_missing_first_slot_rejected(self):
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
        ]
        ranges = [SlotRange(1, 16383, "node-1", source="shard-1")]
        with pytest.raises(TopologyCompileError, match="not fully covered"):
            compile_topology(nodes, ranges)

    def test_missing_last_slot_rejected(self):
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
        ]
        ranges = [SlotRange(0, 16382, "node-1", source="shard-1")]
        with pytest.raises(TopologyCompileError, match="not fully covered"):
            compile_topology(nodes, ranges)


class TestReplicaDoesNotOwnSlots:
    """replicas do not own slots directly."""

    def test_replica_owning_slots_rejected(self):
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
            TopologyNode("node-2", "127.0.0.1", 7001, role=REPLICA),
        ]
        ranges = [
            SlotRange(0, 8191, "node-1", source="shard-1"),
            SlotRange(8192, 16383, "node-2", source="shard-2"),
        ]
        with pytest.raises(TopologyCompileError, match="replica"):
            compile_topology(nodes, ranges)

    def test_primary_with_replica_no_slots_on_replica(self):
        # a primary + its replica: only the primary owns slots
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
            TopologyNode("node-1r", "127.0.0.1", 7001, role=REPLICA),
        ]
        ranges = [SlotRange(0, 16383, "node-1", source="shard-1")]
        snapshot = compile_topology(nodes, ranges)

        assert set(snapshot.primaries()) == {"node-1"}
        assert set(snapshot.replicas()) == {"node-1r"}
        for slot in range(REDIS_CLUSTER_HASH_SLOTS):
            assert snapshot.owner_node_id(slot) == "node-1"


class TestNodeIdAndAddressSeparate:
    """node ID and address are separate identities."""

    def test_same_address_distinct_node_ids(self):
        # two nodes share an address but have distinct node IDs
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
            TopologyNode("node-2", "127.0.0.1", 7000, role=PRIMARY),
        ]
        ranges = [
            SlotRange(0, 8191, "node-1", source="shard-1"),
            SlotRange(8192, 16383, "node-2", source="shard-2"),
        ]
        snapshot = compile_topology(nodes, ranges)
        assert snapshot.owner(0).node_id == "node-1"
        assert snapshot.owner(8192).node_id == "node-2"

    def test_duplicate_node_id_rejected(self):
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
            TopologyNode("node-1", "127.0.0.1", 7001, role=PRIMARY),
        ]
        ranges = [SlotRange(0, 16383, "node-1", source="shard-1")]
        with pytest.raises(TopologyCompileError, match="duplicate node_id"):
            compile_topology(nodes, ranges)

    def test_address_is_separate_from_identity(self):
        node = TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY)
        assert node.node_id == "node-1"
        assert node.address == ("127.0.0.1", 7000)


class TestUnknownNodeRejected:
    """Unknown nodes are rejected, not auto-fixed."""

    def test_unknown_node_rejected(self):
        nodes = [
            TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY),
        ]
        ranges = [
            SlotRange(0, 8191, "node-1", source="shard-1"),
            SlotRange(8192, 16383, "node-ghost", source="shard-2"),
        ]
        with pytest.raises(TopologyCompileError, match="unknown node_id"):
            compile_topology(nodes, ranges)


class TestConnectCountIsZero:
    """the compiler never connects to Redis."""

    def test_connect_count_is_zero(self):
        nodes, ranges = _two_node_topology()
        with patch("redis.connection.Connection.connect") as connect_mock:
            snapshot = compile_topology(nodes, ranges)
            assert connect_mock.call_count == 0
        # the snapshot is usable without any connection
        assert snapshot.owner(0).node_id == "node-1"

    def test_no_redis_client_created(self):
        nodes, ranges = _two_node_topology()
        with patch("redis.client.Redis") as redis_mock:
            snapshot = compile_topology(nodes, ranges)
            assert redis_mock.call_count == 0
        assert snapshot.slot_count == REDIS_CLUSTER_HASH_SLOTS


class TestInputValidation:
    """Invalid inputs are rejected with clear errors."""

    def test_slot_range_out_of_bounds(self):
        nodes = [TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY)]
        with pytest.raises(TopologyCompileError, match="outside"):
            compile_topology(nodes, [SlotRange(-1, 100, "node-1")])
        with pytest.raises(TopologyCompileError, match="outside"):
            compile_topology(nodes, [SlotRange(0, 16384, "node-1")])

    def test_slot_range_start_greater_than_end(self):
        nodes = [TopologyNode("node-1", "127.0.0.1", 7000, role=PRIMARY)]
        with pytest.raises(TopologyCompileError, match="greater than end"):
            compile_topology(nodes, [SlotRange(100, 50, "node-1")])

    def test_invalid_role(self):
        with pytest.raises(TopologyCompileError, match="role"):
            TopologyNode("node-1", "127.0.0.1", 7000, role="invalid")

    def test_empty_node_id(self):
        with pytest.raises(TopologyCompileError, match="node_id"):
            TopologyNode("", "127.0.0.1", 7000, role=PRIMARY)

    def test_owner_out_of_bounds(self):
        nodes, ranges = _two_node_topology()
        snapshot = compile_topology(nodes, ranges)
        with pytest.raises(TopologyCompileError, match="slot must be in"):
            snapshot.owner(-1)
        with pytest.raises(TopologyCompileError, match="slot must be in"):
            snapshot.owner(16384)


class TestImmutableSnapshot:
    """The snapshot is immutable after construction."""

    def test_snapshot_cannot_be_mutated(self):
        nodes, ranges = _two_node_topology()
        snapshot = compile_topology(nodes, ranges)
        # internal mappings are tuples, not lists
        assert isinstance(snapshot._slot_owners, tuple)
        assert isinstance(snapshot._range_sources, tuple)
        # the public accessors return copies; mutating them does not affect
        # the snapshot
        pub_nodes = snapshot.nodes()
        pub_nodes["new"] = TopologyNode("new", "127.0.0.1", 7002)
        assert "new" not in snapshot.nodes()
        pub_primaries = snapshot.primaries()
        pub_primaries["new"] = TopologyNode("new", "127.0.0.1", 7002, role=PRIMARY)
        assert "new" not in snapshot.primaries()
