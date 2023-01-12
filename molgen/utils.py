"""Some utility functions"""

import torch


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
