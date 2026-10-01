"""Tests for MOVED/ASK redirect tracking."""
import pytest
from unittest import mock
from redis.asyncio.cluster import ClusterNode, RedirectTracker
from redis.exceptions import AskError, MovedError
from .test_cluster import default_cluster_slots, default_host, get_mocked_redis_client

class TestRedirectTracker:
    """Single-request redirect tracking for the async cluster client."""

    @pytest.mark.asyncio
    async def test_moved_ask_circular_redirects_report(self) -> None:
        """Verify final response and report for MOVED/ASK/circular redirects."""
        r = await get_mocked_redis_client(host=default_host, port=7000, cluster_slots=default_cluster_slots)
        key = "bar"
        slot = r.keyslot(key)
        assert slot < 8192
        nodes = list(r.get_nodes())
        assert len(nodes) >= 4
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

        with mock.patch.object(ClusterNode, "execute_command", autospec=True, side_effect=execute_command):
            tracker = RedirectTracker()
            result = await r.execute_command("GET", key, redirect_tracker=tracker)

        assert result == "final_value"
        report = tracker.report()
        assert report["total_redirects"] == 4
        assert report["moved_count"] == 3
        assert report["ask_count"] == 1
        assert report["circular_count"] == 1

        moved = tracker.moved_events
        assert len(moved) == 3
        assert moved[0]["kind"] == "MOVED"
        assert moved[0]["source_node"] == node0.name
        assert moved[0]["destination_node"] == node0.name
        assert moved[0]["circular"] is True
        assert moved[0]["slot_id"] == slot
        assert moved[1]["source_node"] == node0.name
        assert moved[1]["destination_node"] == node1.name
        assert moved[1]["circular"] is False
        assert moved[1]["slot_id"] == slot
        assert moved[2]["source_node"] == node1.name
        assert moved[2]["destination_node"] == node2.name
        assert moved[2]["circular"] is False
        assert moved[2]["slot_id"] == slot

        ask = tracker.ask_events
        assert len(ask) == 1
        assert ask[0]["kind"] == "ASK"
        assert ask[0]["source_node"] == node2.name
        assert ask[0]["destination_node"] == node3.name
        assert ask[0]["circular"] is False
        assert ask[0]["slot_id"] == slot

        assert all(e["kind"] == "MOVED" for e in moved)
        assert all(e["kind"] == "ASK" for e in ask)
        assert all(e["kind"] != "ASK" for e in moved)
        assert all(e["kind"] != "MOVED" for e in ask)

        report_str = str(report)
        assert "password" not in report_str.lower()
        assert "auth" not in report_str.lower()
        assert "credential" not in report_str.lower()
        assert "token" not in report_str.lower()
        assert all("timestamp" in e for e in report["events"])

    @pytest.mark.asyncio
    async def test_last_redirect_tracker_populated_when_omitted(self) -> None:
        """When no tracker is passed, the client creates one as last_redirect_tracker."""
        r = await get_mocked_redis_client(host=default_host, port=7000, cluster_slots=default_cluster_slots)
        key = "bar"
        slot = r.keyslot(key)
        node0 = r.get_node_from_key(key, False)
        nodes = list(r.get_nodes())
        node1 = next(n for n in nodes if n != node0)

        def execute_command(node, *args, **kwargs):
            if args[0] == "ASKING":
                return "OK"
            if node.name == node0.name:
                raise MovedError(f"{slot} {node1.host}:{node1.port}")
            if node.name == node1.name:
                return "ok"
            return "MOCK_OK"

        with mock.patch.object(ClusterNode, "execute_command", autospec=True, side_effect=execute_command):
            result = await r.execute_command("GET", key)

        assert result == "ok"
        assert r.last_redirect_tracker is not None
        assert r.last_redirect_tracker.moved_count == 1
        assert r.last_redirect_tracker.ask_count == 0
        assert r.last_redirect_tracker.circular_count == 0
        assert r.last_redirect_tracker.moved_events[0]["source_node"] == node0.name
        assert r.last_redirect_tracker.moved_events[0]["destination_node"] == node1.name

    @pytest.mark.asyncio
    async def test_no_redirects_report_is_empty(self) -> None:
        """A request that follows no redirects produces an empty report."""
        r = await get_mocked_redis_client(host=default_host, port=7000, cluster_slots=default_cluster_slots)

        def execute_command(node, *args, **kwargs):
            return "ok"

        with mock.patch.object(ClusterNode, "execute_command", autospec=True, side_effect=execute_command):
            tracker = RedirectTracker()
            result = await r.execute_command("GET", "bar", redirect_tracker=tracker)

        assert result == "ok"
        report = tracker.report()
        assert report["total_redirects"] == 0
        assert report["moved_count"] == 0
        assert report["ask_count"] == 0
        assert report["circular_count"] == 0
        assert report["events"] == []
