from pathlib import Path
import pickle  # nosec B403

import pytest
import torch

from lit_wsl.mapper.weight_mapper import ParameterInfo, WeightMapper


@pytest.mark.skipif(
    not Path("data/source_params.pkl").exists() or not Path("data/target_params.pkl").exists(),
    reason="Required model parameter files are missing.",
)
def test_yolo_model_integration():
    with Path("data/source_params.pkl").open("rb") as f:
        source_params = pickle.load(f)  # noqa: S301  # nosec B301

    with Path("data/target_params.pkl").open("rb") as f:
        target_params = pickle.load(f)  # noqa: S301  # nosec B301

    dummy_input = torch.zeros((1, 3, 640, 640))
    mapper = WeightMapper(source_params=source_params, target_params=target_params, dummy_input=dummy_input)
    result = mapper.suggest_mapping(threshold=0.3)

    mapping = result.get_mapping()

    unmatched = result.get_unmatched()

    assert mapping is not None
    assert isinstance(unmatched, dict)
