from typing import Self

from lit_wsl.mapper.parameter_group import ParameterGroup


class ModuleNode:
    """Represents a node in the hierarchical module structure."""

    def __init__(self, name: str, full_path: str, depth: int) -> None:
        """Initialize a module node.

        Args:
            name: The module name (e.g., 'conv1')
            full_path: The full module path (e.g., 'encoder.layer1.conv1')
            depth: Depth in the hierarchy (0 for root)
        """
        self.name = name
        self.full_path = full_path
        self.depth = depth
        self.children: dict[str, ModuleNode] = {}
        self.parent: ModuleNode | None = None
        self.parameter_group: ParameterGroup | None = None

    def add_child(self, child: Self) -> None:
        """Add a child node."""
        self.children[child.name] = child
        child.parent = self

    def get_descendant_groups(self) -> list[ParameterGroup]:
        """Get all parameter groups in this subtree."""
        groups = []
        if self.parameter_group is not None:
            groups.append(self.parameter_group)
        for child in self.children.values():
            groups.extend(child.get_descendant_groups())
        return groups

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (has no children)."""
        return len(self.children) == 0

    def __repr__(self) -> str:
        param_info = f", params={len(self.parameter_group.param_types)}" if self.parameter_group else ""
        return f"ModuleNode('{self.full_path}', depth={self.depth}{param_info})"
