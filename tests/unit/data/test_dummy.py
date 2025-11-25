import pytest
import torch
from torch.utils.data import DataLoader

from lit_wsl.data.dummy import DummyDataset
from lit_wsl.training.stages import Stage


class TestDummyDataset:
    """Test suite for DummyDataset class."""

    def test_init_with_valid_parameters(self) -> None:
        """Test DummyDataset initialization with valid parameters."""
        ds = DummyDataset(Stage.Training, (1, 28, 28), (1,), num_samples=100)
        assert ds.stage == Stage.Training
        assert ds.shapes == ((1, 28, 28), (1,))
        assert ds.num_samples == 100

    def test_init_with_default_num_samples(self) -> None:
        """Test DummyDataset initialization with default num_samples."""
        ds = DummyDataset(Stage.Training, (10,))
        assert ds.num_samples == 1000

    def test_init_with_invalid_num_samples(self) -> None:
        """Test DummyDataset raises ValueError for invalid num_samples."""
        with pytest.raises(ValueError, match="Provide an argument greater than 0 for `num_samples`"):
            DummyDataset(Stage.Training, (10,), num_samples=0)

        with pytest.raises(ValueError, match="Provide an argument greater than 0 for `num_samples`"):
            DummyDataset(Stage.Training, (10,), num_samples=-1)

    def test_len(self) -> None:
        """Test __len__ returns correct number of samples."""
        num_samples = 42
        ds = DummyDataset(Stage.Training, (10,), num_samples=num_samples)
        assert len(ds) == num_samples

    def test_getitem_single_shape(self) -> None:
        """Test __getitem__ with a single shape."""
        shape = (3, 32, 32)
        ds = DummyDataset(Stage.Training, shape, num_samples=5)

        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 1
        assert isinstance(sample[0], torch.Tensor)
        assert sample[0].shape == shape

    def test_getitem_multiple_shapes(self) -> None:
        """Test __getitem__ with multiple shapes."""
        shapes = ((1, 28, 28), (1,), (10,))
        ds = DummyDataset(Stage.Training, *shapes, num_samples=5)

        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 3

        for i, shape in enumerate(shapes):
            assert isinstance(sample[i], torch.Tensor)
            assert sample[i].shape == shape

    def test_getitem_returns_random_tensors(self) -> None:
        """Test that __getitem__ returns random tensors."""
        ds = DummyDataset(Stage.Training, (10,), num_samples=2)

        sample1 = ds[0]
        sample2 = ds[1]

        # Different indices should return different random tensors
        assert not torch.allclose(sample1[0], sample2[0])

    def test_getitem_all_values_in_valid_range(self) -> None:
        """Test that generated tensors have values in [0, 1) range."""
        ds = DummyDataset(Stage.Training, (100,), num_samples=10)

        for i in range(len(ds)):
            sample = ds[i]
            tensor = sample[0]
            assert torch.all(tensor >= 0.0)
            assert torch.all(tensor < 1.0)

    def test_with_dataloader_single_shape(self) -> None:
        """Test DummyDataset works correctly with DataLoader."""
        ds = DummyDataset(Stage.Training, (1, 28, 28), num_samples=100)
        dl = DataLoader(ds, batch_size=7)

        batch = next(iter(dl))
        assert isinstance(batch, list)
        assert len(batch) == 1
        assert batch[0].shape == torch.Size([7, 1, 28, 28])

    def test_with_dataloader_multiple_shapes(self) -> None:
        """Test DummyDataset with DataLoader and multiple shapes."""
        ds = DummyDataset(Stage.Training, (1, 28, 28), (1,), num_samples=100)
        dl = DataLoader(ds, batch_size=7)

        batch = next(iter(dl))
        assert isinstance(batch, list)
        assert len(batch) == 2
        assert batch[0].shape == torch.Size([7, 1, 28, 28])
        assert batch[1].shape == torch.Size([7, 1])

    def test_different_stages(self) -> None:
        """Test DummyDataset can be created for different stages."""
        for stage in [Stage.Training, Stage.Validating, Stage.Testing, Stage.Predicting]:
            ds = DummyDataset(stage, (10,), num_samples=10)
            assert ds.stage == stage
            assert len(ds) == 10

    def test_iterate_full_dataset(self) -> None:
        """Test iterating through the full dataset."""
        num_samples = 10
        ds = DummyDataset(Stage.Training, (5,), num_samples=num_samples)

        count = 0
        for i in range(len(ds)):
            sample = ds[i]
            assert isinstance(sample, tuple)
            assert len(sample) == 1
            assert sample[0].shape == (5,)
            count += 1

        assert count == num_samples

    def test_complex_shapes(self) -> None:
        """Test DummyDataset with complex tensor shapes."""
        shapes = ((3, 224, 224), (512, 7, 7), (1000,))
        ds = DummyDataset(Stage.Training, *shapes, num_samples=5)

        sample = ds[0]
        assert len(sample) == 3
        assert sample[0].shape == (3, 224, 224)
        assert sample[1].shape == (512, 7, 7)
        assert sample[2].shape == (1000,)

    def test_empty_shapes_tuple(self) -> None:
        """Test DummyDataset with no shapes provided."""
        ds = DummyDataset(Stage.Training, num_samples=10)

        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 0

    def test_scalar_shape(self) -> None:
        """Test DummyDataset with scalar (0-dimensional) tensors."""
        ds = DummyDataset(Stage.Training, (), num_samples=5)

        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 0
