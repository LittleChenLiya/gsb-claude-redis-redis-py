"""Offline tests for the explicit topology snapshot compiler.

No Redis server is involved. Local command:

    python -m pytest tests/test_topology.py -q
"""

import socket
from types import MappingProxyType

import pytest
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.topology import (
    ROLE_MASTER,
    ROLE_REPLICA,
    Node,
    SlotRange,
    TopologyCompileError,
    compile_topology,
)

TWO_MASTER_HALVES = [
    SlotRange(0, 8191, "node-a", source="topology.conf:1"),
    SlotRange(8192, 16383, "node-b", source="topology.conf:2"),
]


def two_nodes():
    return [
        Node("node-a", "10.0.0.1", 7000, ROLE_MASTER),
        Node("node-b", "10.0.0.2", 7000, ROLE_MASTER),
    ]


@pytest.mark.fixed_client
class TestCompileAcceptance:
    """The acceptance scenarios from the task."""

    def test_two_halves_cover_all_slots(self):
        snapshot = compile_topology(two_nodes(), TWO_MASTER_HALVES)
        assert snapshot.slot_count == REDIS_CLUSTER_HASH_SLOTS
        assert len(snapshot.slot_owners) == REDIS_CLUSTER_HASH_SLOTS
        assert set(snapshot.slot_owners[:8192]) == {"node-a"}
        assert set(snapshot.slot_owners[8192:]) == {"node-b"}
        # Exact, once-only coverage.
        assert sorted(set(snapshot.slot_owners)) == ["node-a", "node-b"]

    def test_edge_shared_between_ranges_is_overlap(self):
        # The second half starts at the first half's last slot.
        ranges = [
            SlotRange(0, 8192, "node-a"),
            SlotRange(8192, 16383, "node-b"),
        ]
        with pytest.raises(TopologyCompileError) as info:
            compile_topology(two_nodes(), ranges)
        assert any("overlaps" in msg for msg in info.value.errors)
        assert any("8192" in msg for msg in info.value.errors)

    def test_missing_one_slot_is_rejected(self):
        # A one-slot gap in the middle and a one-slot shortfall at the end.
        ranges = [
            SlotRange(0, 8191, "node-a"),
            SlotRange(8193, 16383, "node-b"),
        ]
        with pytest.raises(TopologyCompileError) as info:
            compile_topology(two_nodes(), ranges)
        assert any("8192..8192 uncovered" in msg for msg in info.value.errors)

        tail = [
            SlotRange(0, 8191, "node-a"),
            SlotRange(8192, 16382, "node-b"),
        ]
        with pytest.raises(TopologyCompileError) as info:
            compile_topology(two_nodes(), tail)
        assert any("16383..16383 are missing" in msg for msg in info.value.errors)

    def test_hash_tag_team_a_and_team_b_share_a_slot(self):
        snapshot = compile_topology(two_nodes(), TWO_MASTER_HALVES)
        slot_a = key_slot(b"{team}a")
        slot_b = key_slot(b"{team}b")
        assert slot_a == slot_b
        assert snapshot.owner_for_key(b"{team}a") is snapshot.owner_for_key(b"{team}b")

    def test_no_connections_are_made(self, monkeypatch):
        # The compiler must not open a single socket: no Redis connection,
        # no discovery. Count socket creation while compiling.
        created = []

        real_init = socket.socket.__init__

        def counting_init(self, *args, **kwargs):
            created.append(args)
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "__init__", counting_init)
        compile_topology(two_nodes(), TWO_MASTER_HALVES)
        assert created == []

    def test_replica_does_not_own_slots(self):
        nodes = two_nodes() + [
            Node("node-a-replica", "10.0.0.3", 7000, ROLE_REPLICA, "node-a")
        ]
        snapshot = compile_topology(nodes, TWO_MASTER_HALVES)
        # The replica exists for reporting but never appears as an owner.
        assert snapshot.replicas_of("node-a")[0].node_id == "node-a-replica"
        assert all(owner != "node-a-replica" for owner in snapshot.slot_owners)

        stolen = [
            SlotRange(0, 8191, "node-a"),
            SlotRange(8192, 16383, "node-a-replica"),
        ]
        with pytest.raises(TopologyCompileError) as info:
            compile_topology(nodes, stolen)
        assert any("replica" in msg for msg in info.value.errors)


@pytest.mark.fixed_client
class TestRangeAndSourceReport:
    def test_every_range_source_is_reported(self):
        snapshot = compile_topology(two_nodes(), TWO_MASTER_HALVES)
        assert snapshot.range_report() == (
            (0, 8191, "node-a", "topology.conf:1"),
            (8192, 16383, "node-b", "topology.conf:2"),
        )
        assert snapshot.range_source(0) == "topology.conf:1"
        assert snapshot.range_source(8191) == "topology.conf:1"
        assert snapshot.range_source(8192) == "topology.conf:2"
        assert snapshot.range_source(16383) == "topology.conf:2"

    def test_full_coverage_with_three_parts(self):
        nodes = [Node(f"n{i}", "10.0.0.1", 7000 + i) for i in range(3)]
        ranges = [
            SlotRange(0, 5000, "n0", "shard-0"),
            SlotRange(5001, 11000, "n1", "shard-1"),
            SlotRange(11001, 16383, "n2", "shard-2"),
        ]
        snapshot = compile_topology(nodes, ranges)
        assert snapshot.owner_of_slot(5000).node_id == "n0"
        assert snapshot.owner_of_slot(5001).node_id == "n1"
        assert snapshot.owner_of_slot(11001).node_id == "n2"
        assert snapshot.range_of_slot(11000).source == "shard-1"


@pytest.mark.fixed_client
class TestNodeAndRoleConsistency:
    def test_identity_and_address_are_separate(self):
        # Same node id at a different address is a conflict, while the
        # same address can never serve two ids.
        duplicate_id = two_nodes()
        duplicate_id[1] = Node("node-a", "10.0.0.9", 7000)
        with pytest.raises(TopologyCompileError) as info:
            compile_topology(duplicate_id, TWO_MASTER_HALVES)
        assert any("duplicate node_id" in msg for msg in info.value.errors)

        duplicate_addr = two_nodes()
        duplicate_addr[1] = Node("node-c", "10.0.0.1", 7000)
        with pytest.raises(TopologyCompileError) as info:
            compile_topology(duplicate_addr, TWO_MASTER_HALVES)
        assert any("already used" in msg for msg in info.value.errors)

    def test_unknown_owner_is_not_auto_repaired(self):
        ranges = [
            SlotRange(0, 8191, "ghost"),
            SlotRange(8192, 16383, "node-b"),
        ]
        with pytest.raises(TopologyCompileError) as info:
            compile_topology(two_nodes(), ranges)
        assert any(
            "ghost" in msg and "not declared" in msg for msg in info.value.errors
        )

    def test_replica_of_unknown_master_rejected(self):
        nodes = two_nodes() + [Node("node-c", "10.0.0.3", 7000, ROLE_REPLICA, "nobody")]
        with pytest.raises(TopologyCompileError):
            compile_topology(nodes, TWO_MASTER_HALVES)

    def test_replica_chaining_rejected(self):
        nodes = two_nodes() + [
            Node("node-a-replica", "10.0.0.3", 7000, ROLE_REPLICA, "node-a"),
            Node("node-a-replica2", "10.0.0.4", 7000, ROLE_REPLICA, "node-a-replica"),
        ]
        with pytest.raises(TopologyCompileError) as info:
            compile_topology(nodes, TWO_MASTER_HALVES)
        assert any("not a master" in msg for msg in info.value.errors)

    def test_master_naming_other_master_rejected(self):
        nodes = [
            Node("node-a", "10.0.0.1", 7000, ROLE_MASTER),
            Node("node-b", "10.0.0.2", 7000, ROLE_MASTER, "node-a"),
        ]
        with pytest.raises(TopologyCompileError):
            compile_topology(nodes, TWO_MASTER_HALVES)

    def test_replica_without_master_id_rejected(self):
        nodes = two_nodes() + [Node("node-c", "10.0.0.3", 7000, ROLE_REPLICA, None)]
        with pytest.raises(TopologyCompileError):
            compile_topology(nodes, TWO_MASTER_HALVES)


@pytest.mark.fixed_client
class TestSnapshotImmutability:
    def test_snapshot_indexes_are_immutable(self):
        snapshot = compile_topology(two_nodes(), TWO_MASTER_HALVES)
        assert isinstance(snapshot.nodes, MappingProxyType)
        assert isinstance(snapshot.addresses, MappingProxyType)
        with pytest.raises(TypeError):
            snapshot.slot_owners[0] = "node-b"  # type: ignore[index]
        with pytest.raises(TypeError):
            snapshot.ranges[0] = TWO_MASTER_HALVES[1]  # type: ignore[index]

    def test_address_lookup_is_independent_of_ownership(self):
        snapshot = compile_topology(two_nodes(), TWO_MASTER_HALVES)
        assert snapshot.addresses[("10.0.0.1", 7000)] == "node-a"
        owner = snapshot.owner_of_slot(0)
        assert owner.node_id == "node-a"
        assert owner.address == ("10.0.0.1", 7000)

    def test_byte_keys_only(self):
        snapshot = compile_topology(two_nodes(), TWO_MASTER_HALVES)
        with pytest.raises(TypeError):
            snapshot.owner_for_key("{team}a")  # type: ignore[arg-type]

    def test_slot_bounds(self):
        snapshot = compile_topology(two_nodes(), TWO_MASTER_HALVES)
        with pytest.raises(ValueError):
            snapshot.owner_of_slot(-1)
        with pytest.raises(ValueError):
            snapshot.owner_of_slot(16384)
