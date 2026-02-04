import torch


class ParameterInfo:
    """Metadata container for a model parameter."""

    def __init__(
        self,
        name: str,
        tensor: torch.Tensor,
        execution_order: int | None = None,
        is_buffer: bool = False,
        requires_grad: bool | None = None,
    ) -> None:
        """Initialize parameter metadata.

        Args:
            name: Full parameter name (e.g., 'backbone.layer1.conv.weight')
            tensor: The parameter tensor
            execution_order: Order in which the parent module was executed (for call order tracking)
            is_buffer: Whether this is a buffer (running_mean, running_var, etc.) vs trainable parameter
            requires_grad: Whether this parameter requires gradients (auto-detected if None)
        """
        self.name = name
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.numel = tensor.numel()
        self.execution_order = execution_order

        # Buffer and gradient tracking
        self.is_buffer = is_buffer
        self.requires_grad = requires_grad if requires_grad is not None else tensor.requires_grad

        # Parse hierarchical structure
        self.parts = name.split(".")
        self.depth = len(self.parts)
        self.param_name = self.parts[-1]  # e.g., 'weight', 'bias'
        self.module_path = ".".join(self.parts[:-1]) if self.depth > 1 else ""

        # Extract tokens for name matching
        self.tokens = self._extract_tokens(name)

    @property
    def is_trainable(self) -> bool:
        """Check if this parameter is trainable (requires gradients)."""
        # Handle backward compatibility
        is_buf = getattr(self, "is_buffer", False)
        req_grad = getattr(self, "requires_grad", True)
        return req_grad and not is_buf

    @property
    def is_statistical_buffer(self) -> bool:
        """Check if this is a statistical buffer (running stats, tracking counters)."""
        # Handle backward compatibility for pickled objects without is_buffer attribute
        if not hasattr(self, "is_buffer"):
            # Infer from parameter name
            param_name = self.param_name if hasattr(self, "param_name") else self.name.split(".")[-1]
            return param_name in {"running_mean", "running_var", "num_batches_tracked"}

        statistical_names = {"running_mean", "running_var", "num_batches_tracked"}
        return self.is_buffer and self.param_name in statistical_names

    @property
    def is_empty_or_scalar(self) -> bool:
        """Check if this tensor is empty or scalar (potentially ignorable)."""
        return self.numel == 0 or len(self.shape) == 0 or self.shape == (1,)

    def _extract_tokens(self, name: str) -> set[str]:
        """Extract meaningful tokens from parameter name.

        Args:
            name: Parameter name

        Returns:
            Set of tokens
        """
        # Split by common separators and extract meaningful parts
        import re

        # Split on dots, underscores, and camelCase boundaries
        tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", name.replace(".", "_"))
        return {t.lower() for t in tokens if len(t) > 1}

    def __repr__(self) -> str:
        buffer_flag = " [buffer]" if self.is_buffer else ""
        return f"ParameterInfo(name='{self.name}', shape={self.shape}{buffer_flag})"
