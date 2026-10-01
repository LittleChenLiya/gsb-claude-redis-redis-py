"""
Minimal example: single-request MOVED/ASK redirect tracking.

Uses mocked cluster nodes (no real Redis server, no Docker) to simulate
a redirect chain and print the tracking report.

Run: python examples/redirect_tracker_example.py
"""

import asyncio
from unittest import mock

from redis.asyncio.cluster import ClusterNode, RedirectTracker
from redis.exceptions import AskError, MovedError
from tests.test_asyncio.test_cluster import (
    default_cluster_slots,
    default_host,
    get_mocked_redis_client,
)


async def main():
    r = await get_mocked_redis_client(
        host=default_host, port=7000, cluster_slots=default_cluster_slots
    )

    key = "bar"
    slot = r.keyslot(key)
    assert slot < 8192

    nodes = list(r.get_nodes())
    node0 = r.get_node_from_key(key, False)
    node1 = next(n for n in nodes if n != node0)
    node2 = next(n for n in nodes if n not in (node0, node1))
    node3 = next(n for n in nodes if n not in (node0, node1, node2))

    call_counts = {}

    def execute_command(node, *args, **kwargs):
        command = args[0]
        if command == "ASKING":
            return "OK"
        node_name = node.name
        call_counts[node_name] = call_counts.get(node_name, 0) + 1
        count = call_counts[node_name]
        if node_name == node0.name:
            if count == 1:
                raise MovedError(f"{slot} {node0.host}:{node0.port}")
            elif count == 2:
                raise MovedError(f"{slot} {node1.host}:{node1.port}")
        elif node_name == node1.name:
            if count == 1:
                raise MovedError(f"{slot} {node2.host}:{node2.port}")
        elif node_name == node2.name:
            if count == 1:
                raise AskError(f"{slot} {node3.host}:{node3.port}")
        elif node_name == node3.name:
            if count == 1:
                return "final_value"
        return "MOCK_OK"

    with mock.patch.object(
        ClusterNode, "execute_command", autospec=True, side_effect=execute_command
    ):
        tracker = RedirectTracker()
        result = await r.execute_command("GET", key, redirect_tracker=tracker)

    report = tracker.report()
    print("=" * 60)
    print("Redirect tracking example")
    print("=" * 60)
    print(f"Final response: {result!r}")
    print(f"Tracker: {tracker}")
    print()
    print("Report summary:")
    for k, v in report.items():
        if k != "events":
            print(f"  {k}: {v}")
    print()
    print("Events:")
    for i, event in enumerate(report["events"], 1):
        kind = event["kind"]
        src = event["source_node"]
        dst = event["destination_node"]
        slot = event["slot_id"]
        circ = event["circular"]
        print(f"  {i}. kind={kind} src={src} dst={dst} slot={slot} circular={circ}")
    print()
    print("ASK vs MOVED separation:")
    print(f"  MOVED events: {len(tracker.moved_events)}")
    print(f"  ASK events:   {len(tracker.ask_events)}")
    print(f"  Circular MOVED: {len(tracker.circular_events)}")
    print()
    print("No credentials in report:")
    report_str = str(report)
    for secret in ("password", "auth", "credential", "token"):
        print(f"  contains {secret!r}: {secret in report_str.lower()}")


if __name__ == "__main__":
    asyncio.run(main())
