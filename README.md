# lit-wsl

[![Build](https://github.com/twsl/lit-wsl/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/lit-wsl/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/lit-wsl/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/lit-wsl/actions/workflows/docs.yaml)
![GitHub Release](https://img.shields.io/github/v/release/twsl/lit-wsl?include_prereleases)
[![PyPI - Package Version](https://img.shields.io/pypi/v/lit-wsl?logo=pypi&style=flat&color=orange)](https://pypi.org/project/lit-wsl/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lit-wsl?logo=pypi&style=flat&color=blue)](https://pypi.org/project/lit-wsl/)
[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/twsl/lit-wsl/releases)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

My personal library of reusable Pytorch Lightning components

## Features

- [IntermediateLayerGetter](./src/lit_wsl/models/intermediate_layer_getter.py)
- [WeightRenamer](./src/lit_wsl/models/weight_renamer.py)
- [WeightMapper](./src/lit_wsl/mapper/weight_mapper.py)
- [ModelTester](./src/lit_wsl/testing/lightning_tester.py)

## Installation

With `pip`:

```bash
python -m pip install lit-wsl
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv add lit-wsl
```

## How to use it

### IntermediateLayerGetter

Capture intermediate layer outputs during forward pass:

```python
import torch
from torchvision.models import resnet18
from lit_wsl.models.intermediate_layer_getter import IntermediateLayerGetter

model = resnet18(pretrained=True)
# Specify which layers to capture: {layer_name: output_name}
return_layers = {"layer2": "feat1", "layer4": "feat2"}
layer_getter = IntermediateLayerGetter(model, return_layers, keep_output=True)

x = torch.randn(1, 3, 224, 224)
intermediate_outputs, final_output = layer_getter(x)
# intermediate_outputs is OrderedDict with keys "feat1" and "feat2"
print(intermediate_outputs["feat1"].shape)  # torch.Size([1, 128, 28, 28])
```

### WeightRenamer

Rename keys in checkpoint files:

```python
from lit_wsl.models.weight_renamer import WeightRenamer

# Load checkpoint
renamer = WeightRenamer("old_model.pth")

# Remove common prefix
renamer.remove_prefix("model.")

# Rename specific keys
renamer.rename_keys({
    "backbone.conv1": "encoder.conv1",
    "head.fc": "classifier.fc"
})

# Save modified checkpoint
renamer.save("renamed_model.pth")
```

### WeightMapper

Automatically map weights between different model architectures:

```python
import torch
from lit_wsl.mapper.weight_mapper import WeightMapper
from lit_wsl.models.weight_renamer import WeightRenamer

# Define your models (with different layer names)
old_model = OldModelArchitecture()
new_model = NewModelArchitecture()

# Analyze and suggest mapping
mapper = WeightMapper(old_model, new_model)
mapping, unmatched = mapper.suggest_mapping(threshold=0.6)

# Apply mapping to checkpoint
renamer = WeightRenamer("old_weights.pth")
renamer.rename_keys(mapping)
renamer.save("adapted_weights.pth")

# Load adapted weights
new_model.load_state_dict(torch.load("adapted_weights.pth"))
```

## Docs

```bash
uv run mkdocs build -f ./mkdocs.yml -d ./_build/
```

## Update template

```bash
copier update --trust -A --vcs-ref=HEAD
```

## Credits

This project was generated with [![🚀 python project template.](https://img.shields.io/badge/python--project--template-%F0%9F%9A%80-brightgreen)](https://github.com/twsl/python-project-template)
