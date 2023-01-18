"""Some utility functions"""

import torch
import math
from inspect import isfunction
import numpy as np


class MinMaxScaler(torch.nn.Module):
    """A PyTorch module for performing min-max scaling on tensors.
    This module is similar to the `MinMaxScaler` in scikit-learn, and
    implements the `.fit`, `.transform`, and `.inverse_transform` methods.
    Parameters
    ----------
        dim: int
            dimension of features to be scaled
        feature_range: tuple, defulat = (-1, 1)
            The range of the output data. Default is (-1, 1).
    """

    def __init__(self, dim: int, feature_range: tuple = (-1, 1)):
        super().__init__()
        self.feature_range = feature_range
        self.register_buffer("min", torch.empty(dim))
        self.register_buffer("max", torch.empty(dim))
        self.register_buffer("range", torch.empty(dim))
        self.register_buffer("is_fit", torch.tensor([False]).bool())

    def _check_if_fit(self):
        if not self.is_fit:
            raise ValueError("Scaler has not been fit yet. Call .fit first.")

    def forward(self, X):
        self._check_if_fit()
        return self.feature_range[0] + (X - self.min) * self.range

    def fit(self, X):
        """Fit the scaler to the data.
        Parameters
        ----------
            X: torch.Tensor
                A tensor of shape (n, d) where n is the number of data points and d is the number of dimensions.
        Returns
        ----------
            self
        """
        self.min = X.min(dim=0)[0]
        self.max = X.max(dim=0)[0]
        self.range = (self.feature_range[1] - self.feature_range[0]) / (
            self.max - self.min
        )
        self.is_fit = torch.tensor([True]).bool()
        return self

    def transform(self, X):
        """Transform the data using the scaler.
        Parameters
        ----------
            X: torch.Tensor
                A tensor of shape (n, d) where n is the number of data points and d is the number of dimensions.
        Returns
        ----------
            torch.Tensor
                A tensor of shape (n, d) with the scaled data.
        """
        return self.forward(X)

    def inverse_transform(self, X):
        """Inverse transform the scaled data back to the original space.
        Parameters
        ----------
            X: torch.Tensor
                A tensor of shape (n, d) where n is the number of data points and d is the number of dimensions.
        Returns
        ----------
            torch.Tensor
                A tensor of shape (n, d) with the data in the original space.
        """
        self._check_if_fit()
        return self.min + (X - self.feature_range[0]) / self.range


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def linear_schedule(timesteps, s=0.008):
    """
    linear schedule
    """
    betas = np.linspace(0.0001, 0.02, timesteps, dtype=np.float64)
    return np.clip(betas, a_min=0, a_max=0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    betas = betas.numpy()
    return np.clip(betas, a_min=0, a_max=0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def generate_inprint_mask(n_batch, op_num, unmask_index=None):
    """
    The mask will be True where we keep the true value and false where we want to infer the value
    So far it only supporting masking the right side of images
    """

    mask = torch.zeros((n_batch, 1, op_num), dtype=bool)
    # if not unmask_index == None:
    if unmask_index is not None:
        mask[:, :, unmask_index] = True
    return mask


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
