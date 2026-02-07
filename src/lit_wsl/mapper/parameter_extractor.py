from logging import Logger
import warnings

import torch
from torch import nn

from lit_wsl.mapper.parameter_group import ParameterGroup
from lit_wsl.mapper.parameter_info import ParameterInfo
from lit_wsl.utils.logger import get_logger


class ParameterExtractor:
    """Extracts parameter information from PyTorch modules and state dictionaries.

    This class handles:
    - Extracting parameters from nn.Module instances
    - Extracting parameters from state dictionaries (checkpoints)
    - Tracking execution order via forward hooks
    - Grouping parameters by module path
    - Building shape indices for fast lookups
    """

    def __init__(self) -> None:
        """Initialize the ParameterExtractor."""
        self.logger: Logger = get_logger(self.__class__.__name__)

    def extract_from_module(
        self, module: nn.Module, dummy_input: torch.Tensor | None = None
    ) -> dict[str, ParameterInfo]:
        """Extract parameter information from a module.

        Args:
            module: PyTorch module to extract from
            dummy_input: (Optional) Dummy input tensor for execution order tracking.
                        When None (default), no execution order tracking is performed.
                        When provided, runs a forward pass to track layer execution order.

        Returns:
            Dictionary mapping parameter names to ParameterInfo objects
        """
        execution_order_map = {}

        if dummy_input is not None:
            try:
                execution_order_map = self._get_execution_order(module, dummy_input)
            except Exception as e:
                warnings.warn(
                    f"Execution order tracking failed: {e}. Proceeding without execution order information.",
                    stacklevel=2,
                )

        params = {}

        for name, param in module.named_parameters():
            module_path = ".".join(name.split(".")[:-1])
            execution_order = execution_order_map.get(module_path, None)
            params[name] = ParameterInfo(
                name,
                param,
                execution_order,
                is_buffer=False,
                requires_grad=param.requires_grad,
            )

        for name, buffer in module.named_buffers():
            module_path = ".".join(name.split(".")[:-1])
            execution_order = execution_order_map.get(module_path, None)
            params[name] = ParameterInfo(
                name,
                buffer,
                execution_order,
                is_buffer=True,
                requires_grad=buffer.requires_grad,
            )

        return params

    def extract_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, ParameterInfo]:
        """Extract parameter information from a state dictionary.

        Args:
            state_dict: Dictionary mapping parameter names to tensors

        Returns:
            Dictionary mapping parameter names to ParameterInfo objects
        """
        params = {}
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                param_name = name.split(".")[-1]
                is_buffer = param_name in {
                    "running_mean",
                    "running_var",
                    "num_batches_tracked",
                    "running_mu",
                    "running_sigma",
                }

                params[name] = ParameterInfo(
                    name,
                    tensor,
                    execution_order=None,
                    is_buffer=is_buffer,
                    requires_grad=tensor.requires_grad,
                )
        return params

    def _get_execution_order(self, module: nn.Module, dummy_input: torch.Tensor) -> dict[str, int]:
        """Get execution order of modules by running a forward pass.

        Args:
            module: PyTorch module to analyze
            dummy_input: Dummy input tensor to use for forward pass

        Returns:
            Dictionary mapping module paths to execution order indices

        Raises:
            RuntimeError: If forward pass fails with the provided dummy_input
        """
        try:
            module.eval()
            with torch.no_grad():
                test_output = module(dummy_input)
            del test_output
        except Exception as e:
            raise RuntimeError(
                f"Failed to run forward pass with provided dummy_input. "
                f"Please ensure the input shape and type are compatible with the model. "
                f"Error: {e}"
            ) from e

        execution_order = {}
        order_counter = [0]

        hooks = []
        for name, submodule in module.named_modules():
            if name == "":
                continue

            def hook(module, input, output, module_name=name):
                if module_name not in execution_order:
                    execution_order[module_name] = order_counter[0]
                    order_counter[0] += 1

            hooks.append(submodule.register_forward_hook(hook))

        module.eval()
        with torch.no_grad():
            module(dummy_input)

        for h in hooks:
            h.remove()

        return execution_order

    def extract_parameter_groups(self, params: dict[str, ParameterInfo]) -> dict[str, ParameterGroup]:
        """Extract parameter groups from parameters.

        Groups parameters by their module path (e.g., all weight, bias for a layer).

        Args:
            params: Dictionary of parameter information

        Returns:
            Dictionary mapping module paths to ParameterGroup objects
        """
        groups_dict: dict[str, dict[str, ParameterInfo]] = {}

        for _name, info in params.items():
            module_path = info.module_path
            param_type = info.param_name

            if module_path not in groups_dict:
                groups_dict[module_path] = {}

            groups_dict[module_path][param_type] = info

        groups = {}
        for module_path, param_dict in groups_dict.items():
            groups[module_path] = ParameterGroup(module_path, param_dict)

        return groups

    def build_shape_index(self, params: dict[str, ParameterInfo]) -> dict[tuple, list[str]]:
        """Build an index of parameters by shape for fast lookup.

        Args:
            params: Dictionary of parameter information

        Returns:
            Dictionary mapping shapes to list of parameter names
        """
        index = {}
        for param_name, info in params.items():
            if info.shape not in index:
                index[info.shape] = []
            index[info.shape].append(param_name)
        return index

    def build_group_index(self, groups: dict[str, ParameterGroup]) -> dict[frozenset, list[str]]:
        """Build an index of groups by their parameter types.

        Args:
            groups: Dictionary of ParameterGroup objects

        Returns:
            Dictionary mapping frozensets of parameter types to list of module paths
        """
        index: dict[frozenset, list[str]] = {}
        for module_path, group in groups.items():
            param_types = frozenset(group.param_types)
            if param_types not in index:
                index[param_types] = []
            index[param_types].append(module_path)
        return index

    def filter_buffers(self, params: dict[str, ParameterInfo]) -> dict[str, ParameterInfo]:
        """Filter out buffer parameters for performance optimization.

        When using buffer_matching_mode='exclude', this method removes all buffers
        from the parameter dictionary, significantly improving performance by reducing
        the comparison space.

        Args:
            params: Dictionary of parameter information

        Returns:
            Dictionary containing only trainable parameters (no buffers)
        """
        return {name: info for name, info in params.items() if not info.is_buffer}
