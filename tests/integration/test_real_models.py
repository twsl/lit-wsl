from pathlib import Path
import pickle

from lit_wsl.mapper.weight_mapper import ParameterInfo, WeightMapper


def test_ppyoloe_model_integration():
    with Path("data/PPYOLOE/source_params.pkl").open("rb") as f:
        source_params = pickle.load(f)  # noqa: S301

    with Path("data/PPYOLOE/target_params.pkl").open("rb") as f:
        target_params = pickle.load(f)  # noqa: S301

    mapper = WeightMapper(source_params=source_params, target_params=target_params)
    result = mapper.suggest_mapping(threshold=0.3)

    mapping = result.get_mapping()

    unmatched = result.get_unmatched()

    assert mapping is not None
    assert isinstance(unmatched, dict)
