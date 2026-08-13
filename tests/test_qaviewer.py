import asyncio

import pytest
from textual.widgets import Input, Static

from esgf_qa.qaviewer import QCViewer, iter_nodes, transform_keys


# ------------------------
# Unit tests for helpers
# ------------------------
def test_transform_keys_basic():
    data = {
        "info": {"id": "DS1", "date": "2025-10-30"},
        "fail": {3: {"test1": ["file1.nc"]}},
        "pass": {3: {"test2": ["file2.nc"]}},
        "error": {"checker1": {"func": "some error"}},
    }
    result = transform_keys(data)
    assert "Info" in result
    assert "Failed Checks" in result
    assert "Passed Checks" in result
    assert "Runtime Errors" in result
    assert result["Info"]["Dataset-ID"] == "DS1"
    assert result["Failed Checks"]["Required"]["test1"] == ["file1.nc"]


def test_iter_nodes_flat_tree():
    class Node:
        def __init__(self):
            self.children = []

    root = Node()
    root.children = [Node(), Node()]
    nodes = list(iter_nodes(root))
    assert len(nodes) == 3


# ------------------------
# Async tests for QCViewer
# ------------------------
@pytest.mark.asyncio
async def test_toggle_mouse_support(monkeypatch):
    app = QCViewer({"Info": {"Dataset-ID": "DS1"}})
    driver_calls = []

    async with app.run_test() as pilot:
        monkeypatch.setattr(
            app._driver,
            "_disable_mouse_support",
            lambda: driver_calls.append("disabled"),
            raising=False,
        )
        monkeypatch.setattr(
            app._driver,
            "_enable_mouse_support",
            lambda: driver_calls.append("enabled"),
            raising=False,
        )

        app.action_focus_search()
        await pilot.press("f2")
        await pilot.pause()

        assert not app.mouse_enabled
        assert not app._driver._mouse
        assert driver_calls == ["disabled"]
        guidance = str(app.query_one("#mouse-guidance", Static).render())
        assert "Text-selection mode" in guidance
        assert "Ctrl+Shift+C" in guidance
        assert "not Ctrl+C" in guidance
        assert "right-click the selection" in guidance
        assert "F2: disable text selection" in guidance
        assert "Text selection enabled" in str(app.query_one("#status", Static).render())

        await pilot.press("f2")
        await pilot.pause()

        assert app.mouse_enabled
        assert app._driver._mouse
        assert driver_calls == ["disabled", "enabled"]
        guidance = str(app.query_one("#mouse-guidance", Static).render())
        assert "Left-click: toggle node" in guidance
        assert "Right-click: expand/collapse subtree" in guidance
        assert "F2: enable text selection" in guidance


@pytest.mark.asyncio
async def test_qcviewer_tree_population():
    """
    Tests that result.json is correctly converted into the tree widget structure.

    - Starts QCViewer app in test environment
    - Waits for the population of the tree
    - Asserts that the root node has children
    - Asserts that the "info" node exists
    - Asserts that Dataset-ID node exists
    """
    data = {"Info": {"Dataset-ID": "DS1"}, "Failed Checks": {}, "Passed Checks": {}}
    app = QCViewer(data)

    async with app.run_test() as _pilot:
        # wait until tree root has children
        for _ in range(20):  # try up to 2 seconds
            if app.qc_tree.root.children:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Tree did not populate children in time")

        tree = app.qc_tree
        root = tree.root
        assert root.children
        # Get the Info node
        info_node = next((c for c in root.children if str(c.label) == "Info"), None)
        assert info_node is not None
        # Check Dataset-ID child
        ds_id_node = next(
            (c for c in info_node.children if str(c.label) == "Dataset-ID"), None
        )
        assert ds_id_node is not None


@pytest.mark.asyncio
async def test_search_functionality():
    """
    Tests using the functionality to search the tree.

    - Focuses on search input
    - Simulates submission of a search query
    - Asserts that ``matches`` is populated correctly
    - Checks the navigation with ``action_next_match``, ``action_prev_match``
    """
    data = {"Info": {"Dataset-ID": "DS1"}, "Failed Checks": {}, "Passed Checks": {}}
    app = QCViewer(data)

    async with app.run_test() as pilot:
        await pilot.pause()
        search_input = app.query_one("#search", Input)
        app.action_focus_search()

        # Simulate user submitting search text
        app.on_input_submitted(Input.Submitted(search_input, "Dataset-ID"))

        # There should be one match
        assert len(app.matches) == 1
        match_node = app.matches[0]
        assert str(match_node.label) == "Dataset-ID"

        # Test next/prev match wrapping (only one match)
        _old_index = app.match_index
        app.action_next_match()
        assert app.match_index == 0
        app.action_prev_match()
        assert app.match_index == 0


@pytest.mark.asyncio
async def test_toggle_expand_node_behaviour():
    """
    Tests the node expansion/collapse logic.

    - Finds the info node
    - Checks the state (collapsed initially)
    - Calls ``toggle_expand_node``to expand and checks state
    - Calls ``toggle_expand_node`` to collapse and checks state
    """
    data = {"Info": {"Dataset-ID": "DS1", "Date": "2025-10-30"}}
    app = QCViewer(data)

    async with app.run_test() as _pilot:
        # Wait until tree root has children (poll up to 2 seconds)
        for _ in range(20):
            if app.qc_tree.root.children:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Tree did not populate children in time")

        tree = app.qc_tree
        info_node = next(
            (c for c in tree.root.children if str(c.label) == "Info"), None
        )
        assert info_node is not None

        # Initially collapsed
        assert not info_node.is_expanded
        # Expand node
        app.toggle_expand_node(info_node)
        assert info_node.is_expanded
        # Collapse node
        app.toggle_expand_node(info_node)
        assert not info_node.is_expanded
