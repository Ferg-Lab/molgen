import pytest
import torch
from torch.utils.data import DataLoader
from molgen.utils import MinMaxScaler
from molgen.data import GANDataModule

def test_GANDataModule():
    feature_data = torch.rand(10000, 100)
    condition_data = torch.rand(10000, 50)
    batch_size = 1000
    dm = GANDataModule(feature_data, condition_data, batch_size)
    assert isinstance(dm.train_data[0], torch.Tensor)
    assert isinstance(dm.train_data[1], torch.Tensor)
    assert isinstance(dm.feature_scaler, MinMaxScaler)
    assert isinstance(dm.condition_scaler, MinMaxScaler)
    assert dm.x_dim == 100
    assert dm.c_dim == 50
    assert dm.batch_size == 1000
    assert isinstance(dm.train_dataloader(), DataLoader)
    
    # Test for AssertionError when feature_data and condition_data have different number of data points
    feature_data = torch.rand(10000, 100)
    condition_data = torch.rand(1000, 50)
    with pytest.raises(AssertionError):
        dm = GANDataModule(feature_data, condition_data, batch_size)
    
    # Test for AssertionError when feature_data and condition_data have different number of trajectories
    feature_data = [torch.rand(10000, 100) for i in range(3)]
    condition_data = [torch.rand(10000, 50) for i in range(2)]
    with pytest.raises(AssertionError):
        dm = GANDataModule(feature_data, condition_data, batch_size)
    
    # Test for AssertionError when feature_data and condition_data have different types
    feature_data = [torch.rand(10000, 100) for i in range(3)]
    condition_data = torch.rand(10000, 50)
    with pytest.raises(AssertionError):
        dm = GANDataModule(feature_data, condition_data, batch_size)
