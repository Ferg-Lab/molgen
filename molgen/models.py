"""Models for molecular structure generation"""

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from typing import Union
from molgen.modules import (
    SimpleGenerator,
    SimpleDiscriminator,
    GaussianDiffusion,
    Unet1D,
)
from molgen.data import GANDataModule, DDPMDataModule
from molgen.utils import EMA, MinMaxScaler
import copy


class WGANGP(LightningModule):
    """
    A Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) implementation in Pytorch Lightning.

    This class implements a WGAN-GP model, which is a variant of GANs that aims to improve the stability and quality
    of generated samples. The model consists of a generator and a discriminator network, and the objective is to learn
    a generator that can produce samples that are similar to the real data while the discriminator tries to distinguish
    the real and generated samples.

    Parameters
    ----------
    feature_dim : int
        The dimension of the feature space of the data.

    condition_dim : int
        The dimension of the conditional input of the data.

    gen_hidden_dim : int, default = 256
        The dimension of the hidden layers in the generator network

    dis_hidden_dim : int, default = 256
        The dimension of the hidden layers in the discriminator network

    lambda_gp : float, default = 10.0
        The weight of the gradient penalty term in the loss function

    n_critic :int, default = 5
        The number of updates for the discriminator for each update of the generator

    latent_dim : int, default = 128
        The dimension of the noise input to the generator network

    lr : float, default = 5e-5
        The learning rate for the optimizer

    opt : str, default = "rmsprop"
        The optimizer to use

    **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        gen_hidden_dim: int = 256,
        dis_hidden_dim: int = 256,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        latent_dim: int = 128,
        lr: float = 5e-5,
        opt: str = "rmsprop",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = SimpleGenerator(
            latent_dim=self.hparams.latent_dim + self.hparams.condition_dim,
            output_dim=self.hparams.feature_dim,
            hidden_dim=self.hparams.gen_hidden_dim,
        )
        self.discriminator = SimpleDiscriminator(
            output_dim=self.hparams.feature_dim + self.hparams.condition_dim,
            hidden_dim=self.hparams.dis_hidden_dim,
        )

        self._feature_scaler = MinMaxScaler(feature_dim)
        self._condition_scaler = MinMaxScaler(condition_dim)

        self.is_fit = False

    def forward(self, z, c):
        x = self.generator(torch.cat((z, c), dim=-1))
        return x

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples

        alpha = torch.rand((real_samples.size(0), 1), device=real_samples.device)

        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),  # fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, c = batch

        # sample noise
        z = torch.randn(x.size(0), self.hparams.latent_dim, device=x.device)
        z = z.type_as(x)

        # train generator
        if optimizer_idx == 0:

            fake = self(z, c)
            fake = torch.cat((fake, c), dim=-1)
            g_loss = -torch.mean(self.discriminator(fake))

            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            fake = self(z, c)

            # Real images
            real = torch.cat((x, c), dim=-1)
            real_validity = self.discriminator(real)

            # Fake images
            fake = torch.cat((fake, c), dim=-1)
            fake_validity = self.discriminator(fake)

            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real.data, fake.data)

            # Adversarial loss
            d_loss_was = torch.mean(fake_validity) - torch.mean(
                real_validity
            )  # Wasserstein loss
            gp = self.hparams.lambda_gp * gradient_penalty  # gradient penalty
            d_loss = d_loss_was + gp  # full loss

            self.log("d_loss_was", d_loss_was)
            self.log("gp", gp)
            self.log("d_loss", d_loss, prog_bar=True)

            return d_loss

    def configure_optimizers(self):
        opt = self.hparams.opt.lower()
        if opt == "rmsprop":
            opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.hparams.lr)
            opt_d = torch.optim.RMSprop(
                self.discriminator.parameters(), lr=self.hparams.lr
            )
        elif opt == "adam":
            opt_g = torch.optim.Adam(
                self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9)
            )
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9)
            )
        else:
            raise NotImplementedError

        return (
            {"optimizer": opt_g, "frequency": 1},
            {"optimizer": opt_d, "frequency": self.hparams.n_critic},
        )

    def fit(
        self,
        feature_data: Union[torch.Tensor, list],
        condition_data: Union[torch.Tensor, list],
        batch_size: int = 1000,
        max_epochs: int = 100,
        log: Union[str, bool] = False,
        **kwargs,
    ):
        """
        Fit the WGAN on provided data

        Parameters
        ----------
        feature_data : torch.Tensor (single traj) or list[torch.Tensor] (multi traj)
            tensor with dimentions dim 0 = steps, dim 1 = features of features representing the real data that
            is strived to be recapitulated by the generative model

        condition_data : torch.Tensor (single traj) or list[torch.Tensor] (multi traj)
            list of tensors with dimentions dim 0 = steps, dim 1 = features of features representing the conditioning
            variables associated with each data point in feature space

        batch_size : int, default = 1000
            training batch size

        max_epochs : int, default = 100
            maximum number of epochs to train for

        log : str or bool, default = False
            if the results of the training should be logged. If True logs are by default saved in CSV format
            to the directory `./molgen_logs/version_x/`, where `x` increments based on what has been
            logged already. If a string is passed the saving directory is created based on the provided name
            `./molgen_logs/{log}/`

        **kwargs:
            additional keyword arguments to be passed to the the Lightning `Trainer`
        """
        kwargs.get("enable_checkpointing", False)
        datamodule = GANDataModule(
            feature_data=feature_data,
            condition_data=condition_data,
            batch_size=batch_size,
            **kwargs,
        )
        if self.is_fit:
            raise Warning(
                """The `fit` method was called more than once on the same `WGANGP` instance,
                recreating data scaler on dataset from the most recent `fit` invocation. This warning
                can be safely ignored if the `WGANGP` is being fit on the same data"""
            )
        self._feature_scaler = datamodule.feature_scaler
        self._condition_scaler = datamodule.condition_scaler

        if not hasattr(self, "trainer_"):
            self.trainer_ = Trainer(
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                max_epochs=max_epochs,
                logger=False
                if log is False
                else CSVLogger(
                    save_dir="./",
                    name="molgen_logs",
                    version=None if not isinstance(log, str) else log,
                ),
                **kwargs,
            )
            self.trainer_.fit(self, datamodule)
        else:
            self.trainer_.fit(self, datamodule)

        self.is_fit = True
        return self

    def generate(self, c: torch.Tensor):
        """
        Generate samples based on conditioning variables

        Parameters
        ----------
        c : torch.Tensor
            Conditioning variables, float tensor of shape (n_samples, conditioning_dim)

        Returns
        -------
        gen: torch.Tensor
            Generated samples, float tensor of shape (n_samples, feature_dim)

        """

        assert self.is_fit, "model must be fit to data first using `fit`"
        assert (
            c.size(1) == self.hparams.condition_dim
        ), f"inconsistent dimensions, expecting {self.hparams.condition_dim} dim"

        if torch.cuda.is_available():
            self.to("cuda")

        self.eval()
        z = torch.randn(c.size(0), self.hparams.latent_dim, device=self.device)

        with torch.no_grad():
            c = self._condition_scaler.transform(c.to(self.device))
            gen = self.forward(z, c)
            gen = self._feature_scaler.inverse_transform(gen)

        return gen.cpu()

    def save(self, fname: str):
        """
        Generates a synthetic trajectory from an initial starting point `x_0`

        Parameters
        ----------
        fname : str
            file name for saving a model checkpoint
        """

        assert self.is_fit, "model must be fit to data first using `fit`"

        self.trainer_.save_checkpoint(fname)

    def on_load_checkpoint(self, checkpoint):
        self.is_fit = True
        return super().on_load_checkpoint(checkpoint)


class DDPM(LightningModule):
    """
    A Denoising Diffusion Probabilistic Model (DDPM) implementation in Pytorch Lightning.

    Credit: https://github.com/lucidrains/denoising-diffusion-pytorch

    The DDPM class is a PyTorch implementation of the Diffusion Probabilistic Models (DDPM) algorithm for generative modeling.
    It is built on top of the PyTorch Lightning framework and uses the Unet architecture for the generator and the
    GaussianDiffusion class for the diffusion process. The DDPM class also includes an exponential moving average (EMA)
    for stabilizing the training process.


    Parameters
    ----------
    feature_dim : int
        The dimension of the feature space of the data.

    condition_dim : int
        The dimension of the conditional input of the data.

    hidden_dim : int, default = 32
        Hidden dimention of the UNet model

    dis_hidden_dim : int, default = 256
        The dimension of the hidden layers in the discriminator network

    loss_type : str, default = 'l1'
        The type of loss function used in the diffusion process. Acceptable options are
        'l1', 'l2' and 'huber'

    beta_schedule :str, default = 'cosine'
        The schedule for the beta parameter in the diffusion process. Acceptable options are
        'linear', 'cosine' and 'sigmoid'

    timesteps : int, default = 1000
        The number of timesteps in the diffusion process.

    lr : float, default = 2e-5
        The learning rate for the optimizer

    ema_decay : float, default = 0.995
        Decay rate of the EMA

    step_start_ema : int, default = 2000
        The number of steps after which the EMA will begin updating

    update_ema_every : int, default = 10
        The number of steps between updates to the EMA

    **kwargs: Additional keyword arguments passed to the Unet1D model.

    """

    def __init__(
        self,
        feature_dim,
        condition_dim,
        hidden_dim=32,
        loss_type="l1",
        beta_schedule="cosine",
        timesteps=1000,
        lr=2e-5,
        ema_decay=0.995,
        step_start_ema=2000,
        update_ema_every=10,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        model = Unet1D(dim=hidden_dim, **kwargs)
        diffusion = GaussianDiffusion(
            model,
            timesteps=timesteps,
            unmask_number=condition_dim,
            loss_type=loss_type,
            beta_schedule=beta_schedule,
        )

        self.model = diffusion
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)

        self._feature_scaler = MinMaxScaler(feature_dim)
        self._condition_scaler = MinMaxScaler(condition_dim)

        self.reset_parameters()

        self.is_fit = False

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return opt

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.global_step < self.hparams.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def training_step(self, batch, batch_idx):
        loss = self.model(batch[0])
        self.log("loss", loss)
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.global_step % self.hparams.update_ema_every == 0:
            self.step_ema()

    def fit(
        self,
        feature_data: Union[torch.Tensor, list],
        condition_data: Union[torch.Tensor, list],
        batch_size: int = 1000,
        max_epochs: int = 100,
        log: Union[str, bool] = False,
        **kwargs,
    ):
        """
        Fit the DDPM on provided data

        Parameters
        ----------
        feature_data : torch.Tensor (single traj) or list[torch.Tensor] (multi traj)
            tensor with dimentions dim 0 = steps, dim 1 = features of features representing the real data that
            is strived to be recapitulated by the generative model

        condition_data : torch.Tensor (single traj) or list[torch.Tensor] (multi traj)
            list of tensors with dimentions dim 0 = steps, dim 1 = features of features representing the conditioning
            variables associated with each data point in feature space

        batch_size : int, default = 1000
            training batch size

        max_epochs : int, default = 100
            maximum number of epochs to train for

        log : str or bool, default = False
            if the results of the training should be logged. If True logs are by default saved in CSV format
            to the directory `./molgen_logs/version_x/`, where `x` increments based on what has been
            logged already. If a string is passed the saving directory is created based on the provided name
            `./molgen_logs/{log}/`

        **kwargs:
            additional keyword arguments to be passed to the the Lightning `Trainer`
        """
        kwargs.get("enable_checkpointing", False)
        datamodule = DDPMDataModule(
            feature_data=feature_data,
            condition_data=condition_data,
            batch_size=batch_size,
            **kwargs,
        )
        if self.is_fit:
            raise Warning(
                """The `fit` method was called more than once on the same `DDPM` instance,
                recreating data scaler on dataset from the most recent `fit` invocation. This warning
                can be safely ignored if the `DDPM` is being fit on the same data"""
            )
        self._feature_scaler = datamodule.feature_scaler
        self._condition_scaler = datamodule.condition_scaler

        if not hasattr(self, "trainer_"):
            self.trainer_ = Trainer(
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                max_epochs=max_epochs,
                logger=False
                if log is False
                else CSVLogger(
                    save_dir="./",
                    name="molgen_logs",
                    version=None if not isinstance(log, str) else log,
                ),
                **kwargs,
            )
            self.trainer_.fit(self, datamodule)
        else:
            self.trainer_.fit(self, datamodule)

        self.is_fit = True
        return self

    def generate(self, c: torch.Tensor):
        """
        Generate samples based on conditioning variables

        Parameters
        ----------
        c : torch.Tensor
            Conditioning variables, float tensor of shape (n_samples, conditioning_dim)

        Returns
        -------
        gen: torch.Tensor
            Generated samples, float tensor of shape (n_samples, feature_dim)

        """

        assert self.is_fit, "model must be fit to data first using `fit`"
        assert (
            c.size(1) == self.hparams.condition_dim
        ), f"inconsistent dimensions, expecting {self.hparams.condition_dim} dim"

        if torch.cuda.is_available():
            self.to("cuda")

        self.eval()
        c = self._condition_scaler.transform(c.to(self.device)).float().unsqueeze(1)
        c = torch.cat(
            (
                c,
                torch.zeros(
                    c.shape[0],
                    c.shape[1],
                    self.hparams.feature_dim + self.hparams.condition_dim - c.shape[2],
                    dtype=float,
                    device=c.device,
                ),
            ),
            -1,
        ).float()

        gen = self.ema_model.sample(
            self.hparams.feature_dim + self.hparams.condition_dim,
            batch_size=c.shape[0],
            samples=c,
        )
        gen = gen[:, 0, self.hparams.condition_dim :]

        gen = self._feature_scaler.inverse_transform(gen).cpu()

        return gen

    def save(self, fname: str):
        """
        Generates a synthetic trajectory from an initial starting point `x_0`

        Parameters
        ----------
        fname : str
            file name for saving a model checkpoint
        """

        assert self.is_fit, "model must be fit to data first using `fit`"

        self.trainer_.save_checkpoint(fname)

    def on_load_checkpoint(self, checkpoint):
        self.is_fit = True
        return super().on_load_checkpoint(checkpoint)
