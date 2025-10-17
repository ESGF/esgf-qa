import argparse
import json

from textual.app import App, ComposeResult
from textual.events import MouseUp
from textual.widgets import Footer, Header, Input, Static, Tree


def load_json(path):
    with open(path) as f:
        return json.load(f)


def transform_keys(data):
    """Apply similar renaming logic than in display_qc_results.html"""
    weight_map = {3: "Required", 2: "Recommended", 1: "Suggested"}
    result = {}
    info_map = {
        "id": "Dataset-ID",
        "date": "Date",
        "parent_dir": "Root Directory",
        "files": "# Files",
        "datasets": "# Datasets",
        "cc_version": "Compliance Checker Version",
        "checkers": "Applied Checkers",
        "inter_ds_con_checks_ref": "Reference Datasets (inter-dataset consistency checks)",
    }

    if "info" in data and data["info"]:
        result["Info"] = {}
        for key, val in info_map.items():
            result["Info"][val] = data["info"].get(key, "UNSPECIFIED")

    if "error" in data and data["error"]:
        result["Runtime Errors"] = data["error"]

    if "fail" in data and data["fail"]:
        fail_section = {}
        for w, name in weight_map.items():
            fail_section[name] = data["fail"].get(str(w), data["fail"].get(w, {}))
        result["Failed Checks"] = fail_section

    if "pass" in data and data["pass"]:
        pass_section = {}
        for w, name in weight_map.items():
            pass_section[name] = data["pass"].get(str(w), data["pass"].get(w, {}))
        result["Passed Checks"] = pass_section

    return result


def iter_nodes(node):
    """Recursively yield all nodes starting from this one."""
    yield node
    for child in node.children:
        yield from iter_nodes(child)


class QCFooter(Footer):
    def render(self):
        # Get the default legend
        base = super().render()
        # Append custom legend
        return f"{base}  (x| ) Left-click to toggle node  ( |x) Right-click to toggle auto-expansion/collapse"


class QCViewer(App):
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("/", "focus_search", "Search"),
        ("n", "next_match", "Next Match"),
        ("p", "prev_match", "Prev Match"),
    ]

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.matches = []
        self.match_index = -1
        self.qc_tree = None  # will be created in compose()
        self.last_highlighted_node = None

    def compose(self) -> ComposeResult:
        yield Header()
        self.qc_tree = Tree("QC Results", id="qc-tree")
        yield self.qc_tree
        yield Input(placeholder="Search...", id="search")
        yield Static("", id="status")
        yield QCFooter()

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        # Right-click simulation: expand all children
        # Unfortunately, Tree.NodeSelected does not carry button info
        # We rely on a right-click flag set in on_mouse_down
        if getattr(self, "_right_click_pending", False):
            self.toggle_expand_node(event.node)
            self._right_click_pending = False

    def on_mouse_up(self, event: MouseUp):
        # Set a flag if right-click
        if event.button == 3:  # right click
            self._right_click_pending = True
            # node = self.qc_tree.cursor_node
            # if node:
            #    self.toggle_expand_node(node)

    def toggle_expand_node(self, node):
        """Right-click: expand if collapsed, collapse if expanded, recursively."""
        if hasattr(node, "_expanded_state"):
            # toggle previous state
            expanding = not node._expanded_state
        else:
            # first time, check actual state
            expanding = not node.is_expanded

        if not expanding:
            # Collapse node and all children
            self._collapse_tree(node)
        else:
            # Expand node and all children, potentially only up to certain level
            depth = self.get_node_depth(node)
            if depth <= 1:
                self._expand_tree_up_to_depth(node, current_lvl=depth, target_lvl=2)
            else:
                self._expand_tree(node)

    def get_node_depth(self, node):
        """Return depth of node (root=0)."""
        depth = 0
        parent = node.parent
        while parent:
            depth += 1
            parent = parent.parent
        return depth

    def _expand_tree_up_to_depth(self, node, current_lvl, target_lvl):
        """Recursively expand node up to certain depth level."""
        node.expand()
        node._expanded_state = True
        if current_lvl >= target_lvl:
            return
        for child in node.children:
            self._expand_tree_up_to_depth(child, current_lvl + 1, target_lvl)

    def _expand_tree(self, node):
        """Recursively expand node and all children."""
        node.expand()
        node._expanded_state = True
        for child in node.children:
            self._expand_tree(child)

    def _collapse_tree(self, node):
        """Recursively collapses node and all children."""
        for child in node.children:
            self._collapse_tree(child)
        node.collapse()
        node._expanded_state = False

    def on_mount(self):
        self.populate_tree(self.qc_tree.root, self.data)
        self.qc_tree.root.expand()

    def populate_tree(self, node, data):
        if isinstance(data, dict):
            for k, v in data.items():
                child = node.add(k, expand=False)
                self.populate_tree(child, v)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                child = node.add(f"[{i}]", expand=False)
                self.populate_tree(child, v)
        else:
            node.add(repr(data))

    def action_focus_search(self):
        self.query_one("#search").focus()

    def on_input_submitted(self, event: Input.Submitted):
        """Called when the user submits a search in the input."""
        query = event.value.strip()
        self.matches = []
        self.match_index = -1

        if query:
            # Collect all matching nodes
            for node in iter_nodes(self.qc_tree.root):
                if query.lower() in str(node.label).lower():
                    self.matches.append(node)

        if self.matches:
            # Start at first match
            self.match_index = 0
            # Jump to it (expand path, collapse old if needed)
            self.jump_to_match()
            # Return focus to tree so n/p work
            self.set_focus(self.qc_tree)
        else:
            self.query_one("#status", Static).update(f"No matches for '{query}'")

    def focus_match(self):
        node = self.matches[self.match_index]
        node.expand_all()
        self.qc_tree.select_node(node)
        self.qc_tree.scroll_to_node(node)
        self.query_one("#status", Static).update(
            f"Match {self.match_index+1}/{len(self.matches)}: {node.label}"
        )

    def action_next_match(self):
        if self.matches:
            self.match_index = (self.match_index + 1) % len(self.matches)
            self.jump_to_match()

    def action_prev_match(self):
        if self.matches:
            self.match_index = (self.match_index - 1) % len(self.matches)
            self.jump_to_match()

    def jump_to_match(self) -> None:
        """Jump to the current match, expanding its parents and collapsing previous."""
        if not self.matches or self.match_index < 0:
            return

        # Collapse the previously focused node if any
        if hasattr(self, "current_match_node") and self.current_match_node is not None:
            try:
                self.current_match_node.collapse()
            except Exception:
                pass

        # Get the current match node
        node = self.matches[self.match_index]
        self.current_match_node = node

        # Expand all parents so the node is visible
        parent = node.parent
        while parent:
            parent.expand()
            parent = parent.parent

        # Expand this node itself too
        node.expand()

        # Scroll to and select it
        self.qc_tree.select_node(node)
        self.qc_tree.scroll_to_node(node)
        self.qc_tree.select_node(node)

        # Status line
        self.query_one("#status", Static).update(
            f"Match {self.match_index+1}/{len(self.matches)}: {node.label}"
        )


def main():
    parser = argparse.ArgumentParser(description="View QC result JSON files.")
    parser.add_argument(
        "qc_result", metavar="qc_result.json", help="Path to the QC result JSON file."
    )

    args = parser.parse_args()

    data = load_json(args.qc_result)
    transformed = transform_keys(data)

    app = QCViewer(transformed)
    app.run()


if __name__ == "__main__":
    main()
