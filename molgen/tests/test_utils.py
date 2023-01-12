"""Testing the utils"""

import pytest
import torch

from molgen.utils import MinMaxScaler

def test_min_max_scaler():
    # create some random data
    d = 5
    X = torch.randn(100, d)

    # create a scaler and fit it to the data
    scaler = MinMaxScaler(d)
    scaler.fit(X)

    # transform the data
    X_scaled = scaler.transform(X)

    # check that the transformed data is within the specified range
    assert torch.all(X_scaled >= scaler.feature_range[0])
    assert torch.all(X_scaled <= scaler.feature_range[1])

    # inverse transform the data and check that it is equal to the original data
    X_inv = scaler.inverse_transform(X_scaled)
    assert torch.allclose(X, X_inv, rtol=1e-4, atol=1e-5)

    # test that the scaler raises an error if it hasn't been fit yet
    with pytest.raises(ValueError):
        scaler = MinMaxScaler(d)
        scaler.transform(X)