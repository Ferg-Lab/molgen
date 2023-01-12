"""Models for molecular structure generation"""

import torch
from pytorch_lightning import LightningModule, Trainer
from typing import Union
from molgen.modules import (
    SimpleGenerator,
    SimpleDiscriminator,
)
from molgen.data import GANDataModule


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

        **kwargs:
            additional keyword arguments to be passed to the the Lightning `Trainer`
        """
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
                auto_select_gpus=True,
                max_epochs=max_epochs,
                logger=False,
                enable_checkpointing=False,
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
            c.size(1) == self.trainer_.datamodule.c_dim
        ), f"inconsistent dimensions, expecting {self.trainer_.datamodule.c_dim} dim"

        self.eval()
        z = torch.randn(c.size(0), self.hparams.latent_dim, device=self.device)

        with torch.no_grad():
            c = self._condition_scaler.transform(c.to(self.device))
            gen = self.forward(z, c)
            gen = self._feature_scaler.inverse_transform(gen)

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
