"""Two-node example for the explicit Redis Cluster topology compiler.

Declares two masters that split the 16384 slots into exact halves, plus
one replica, then compiles them into an immutable ownership snapshot.
Nothing here connects to Redis.

Run locally::

    python examples/topology_two_nodes.py
"""

from redis.topology import (
    ROLE_MASTER,
    ROLE_REPLICA,
    Node,
    SlotRange,
    TopologyCompileError,
    compile_topology,
)


def build_topology():
    nodes = [
        # node_id (identity) is separate from host/port (address).
        Node(node_id="node-a", host="10.0.0.1", port=7000, role=ROLE_MASTER),
        Node(node_id="node-b", host="10.0.0.2", port=7000, role=ROLE_MASTER),
        # A replica backs node-a but never owns slots itself.
        Node(
            node_id="node-a-replica",
            host="10.0.0.3",
            port=7000,
            role=ROLE_REPLICA,
            master_id="node-a",
        ),
    ]
    ranges = [
        SlotRange(0, 8191, owner_id="node-a", source="cluster-decl.conf: shard-a"),
        SlotRange(8192, 16383, owner_id="node-b", source="cluster-decl.conf: shard-b"),
    ]
    return compile_topology(nodes, ranges)


def main() -> None:
    try:
        snapshot = build_topology()
    except TopologyCompileError as exc:
        print("topology rejected:")
        for problem in exc.errors:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("compiled topology: 0..16383 covered exactly once")
    for start, end, owner_id, source in snapshot.range_report():
        print(f"  slots {start:5d}..{end:5d} -> {owner_id}  [{source}]")

    key = b"{team}a"
    owner = snapshot.owner_for_key(key)
    print(
        f"{key!r} is served by {owner.node_id} at "
        f"{owner.host}:{owner.port} (replica: "
        f"{snapshot.replicas_of('node-a')[0].node_id})"
    )


if __name__ == "__main__":
    main()
