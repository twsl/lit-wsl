from collections import OrderedDict
import functools
from typing import Any

from torch import Tensor, nn


# using wonder's beautiful simplification: https://stackoverflow.com/a/31174427
def rgetattr(obj, attr, *args) -> Any:
    def _getattr(obj, attr) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


# from https://github.com/avishkarsaha/translating-images-into-maps/blob/a6b96425efb97b41f62c58b72c261602c31f2d39/src/model/intermediate_layer_getter.py
class IntermediateLayerGetter:
    def __init__(self, model: nn.Module, return_layers: dict[str, str], keep_output: bool = True) -> None:
        """Wraps a Pytorch module to get intermediate values.

        Arguments:
            model: The Pytorch module to call
            return_layers: Dictionary with the selected submodules
            to return the output (format: {[current_module_name]: [desired_output_name]},
            current_module_name can be a nested submodule, e.g. submodule1.submodule2.submodule3)
            keep_output: If True model_output contains the final model's output

        Returns:
            (mid_outputs {OrderedDict}, model_output {any}) -- mid_outputs keys are
            your desired_output_name (s) and their values are the returned tensors
            of those submodules (OrderedDict([(desired_output_name,tensor(...)), ...).
            See keep_output argument for model_output description.
            In case a submodule is called more than one time, all it's outputs are
            stored in a list.
        """
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output

    def __call__(self, *args, **kwargs) -> tuple[OrderedDict, Any]:
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            layer = rgetattr(self._model, name)

            def hook(module, input, output, new_name=new_name) -> None:
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output)
                    else:
                        ret[new_name] = [ret[new_name], output]
                else:
                    ret[new_name] = output

            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f"Module {name} not found") from e
            handles.append(h)

        output = self._model(*args, **kwargs) if self.keep_output else None

        for h in handles:
            h.remove()

        return ret, output
