from enum import StrEnum


class Stage(StrEnum):
    Training = "train"
    Validating = "val"
    Testing = "test"
    Predicting = "pred"
