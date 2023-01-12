"""Datamodules for molgen"""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Union, List

from molgen.utils import MinMaxScaler


class GANDataModule(LightningDataModule):
    """
    GANDataModule is a Pytorch Lightning DataModule for training GANs.
    It takes in feature_data and condition_data and creates a DataLoader for training. 
    The feature_data and condition_data should be of the same type (either a float tensor for single traj or list of float tensors for multiple trajs)
    and must have the same number of data points.

    Parameters
    ----------
    feature_data : Union[torch.Tensor, List[torch.Tensor]]
        feature data for the GAN, either a float tensor for single traj or list of float tensors for multiple trajs
    
    condition_data : Union[torch.Tensor, List[torch.Tensor]]
        conditioning data for the GAN, either a float tensor for single traj or list of float tensors for multiple trajs
    
    batch_size : int, default = 10000
        batch size for the DataLoader. Default is 1000.

    Attributes
    ----------
    self.feature_scaler: MinMaxScaler
        scaler for scaling the feature data

    self.condition_scaler: MinMaxScaler
        scaler for scaling the conditioning data

    self.x_dim: int
        dimention of the features

    self.c_dim: int
        dimention of the conditioning
    """
    def __init__(
        self,
        feature_data: Union[torch.Tensor, List[torch.Tensor]],
        condition_data: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 1000,
    ):
        super().__init__()
        self.batch_size = batch_size

        assert type(feature_data) == type(
            condition_data
        ), """feature_data and condition_data must be of the same type
        (either a float tensor for single traj or list of float tensors for multiple trajs)"""

        if isinstance(feature_data, torch.Tensor):
            assert (
                feature_data.shape[0] == condition_data.shape[0]
            ), "feature_data and condition_data must have the same number of data points"
        else:
            n_feature = sum([x.shape[0] for x in feature_data])
            n_condition = sum([x.shape[0] for x in condition_data])
            assert (
                n_feature == n_condition
            ), "feature_data and condition_data must have the same number of data points"
            assert len(feature_data) == len(
                condition_data
            ), "feature_data and condition_data must have the same number of trajectories"

        self.feature_scaler = self._get_scaler(feature_data)
        self.condition_scaler = self._get_scaler(condition_data)

        if isinstance(feature_data, torch.Tensor):
            self.train_data = [
                self.feature_scaler.transform(feature_data),
                self.condition_scaler.transform(condition_data),
            ]
        else:
            self.train_data = [
                torch.cat(
                    [self.feature_scaler.transform(x) for x in feature_data], dim=0
                ),
                torch.cat(
                    [self.condition_scaler.transform(x) for x in condition_data], dim=0
                ),
            ]

        self.x_dim = self.train_data[0].shape[1]
        self.c_dim = self.train_data[1].shape[1]

    def _get_scaler(self, data):
        """
        Helper function to get the scaler for the data
        
        Parameters
        ----------
            data : Union[torch.Tensor, List[torch.Tensor]]
                data to be scaled
            
        Returns
        ----------
            MinMaxScaler : Scaler for the data
        """
        if isinstance(data, torch.Tensor):
            d = data.size(1)
            scaler = MinMaxScaler(d)
            scaler.fit(data)
        elif isinstance(data, list):
            d = data[0].size(1)
            scaler = MinMaxScaler(d)
            scaler.fit(torch.cat(data, dim=0))
        else:
            raise TypeError(
                "Data type %s is not supported; must be a float tensor (single traj) or list of float tensors (multi "
                "traj)" % type(data)
            )
        return scaler

    def train_dataloader(self):
        """
        Returns the DataLoader for training the GAN
        
        Returns
        ----------
            DataLoader : Pytorch DataLoader for training the GAN
        """
        dataset = TensorDataset(*self.train_data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
