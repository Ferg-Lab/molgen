import torch
import pytest
from molgen.models import WGANGP, DDPM


@pytest.mark.parametrize(
    "feature_dim,condition_dim,gen_hidden_dim,dis_hidden_dim,lambda_gp,n_critic,latent_dim,lr,opt",
    [
        (100, 10, 256, 256, 10.0, 5, 128, 5e-5, "rmsprop"),
        (100, 10, 256, 256, 10.0, 5, 128, 1e-4, "adam"),
    ],
)
def test_wgangp(
    feature_dim,
    condition_dim,
    gen_hidden_dim,
    dis_hidden_dim,
    lambda_gp,
    n_critic,
    latent_dim,
    lr,
    opt,
):
    model = WGANGP(
        feature_dim,
        condition_dim,
        gen_hidden_dim,
        dis_hidden_dim,
        lambda_gp,
        n_critic,
        latent_dim,
        lr,
        opt,
    )
    assert model.hparams.feature_dim == feature_dim
    assert model.hparams.condition_dim == condition_dim
    assert model.hparams.gen_hidden_dim == gen_hidden_dim
    assert model.hparams.dis_hidden_dim == dis_hidden_dim
    assert model.hparams.lambda_gp == lambda_gp
    assert model.hparams.n_critic == n_critic
    assert model.hparams.latent_dim == latent_dim
    assert model.hparams.lr == lr
    assert model.hparams.opt == opt

    fake_feature_data = torch.randn(10, feature_dim)
    fake_condition_data = torch.randn(10, condition_dim)
    max_epochs = 2

    # test fitting
    model.fit(fake_feature_data, fake_condition_data, max_epochs=max_epochs)

    # test generation
    model.generate(fake_condition_data)


@pytest.mark.parametrize(
    "feature_dim,condition_dim,hidden_dim,loss_type,beta_schedule,timesteps,lr",
    [
        (100, 10, 32, "l1", "linear", 1000, 2e-5),
        (100, 10, 32, "l2", "cosine", 1000, 2e-5),
        (100, 10, 32, "huber", "cosine", 1000, 2e-5),
    ],
)
def test_ddpm(
    feature_dim, condition_dim, hidden_dim, loss_type, beta_schedule, timesteps, lr
):
    model = DDPM(
        feature_dim, condition_dim, hidden_dim, loss_type, beta_schedule, timesteps, lr
    )
    assert model.hparams.feature_dim == feature_dim
    assert model.hparams.condition_dim == condition_dim
    assert model.hparams.hidden_dim == hidden_dim
    assert model.hparams.loss_type == loss_type
    assert model.hparams.beta_schedule == beta_schedule
    assert model.hparams.timesteps == timesteps
    assert model.hparams.lr == lr

    fake_feature_data = torch.randn(10, feature_dim)
    fake_condition_data = torch.randn(10, condition_dim)
    max_epochs = 2

    # test fitting
    model.fit(fake_feature_data, fake_condition_data, max_epochs=max_epochs)

    # test generation
    model.generate(fake_condition_data)
