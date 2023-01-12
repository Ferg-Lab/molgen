import pytest
import torch
from torch import nn
from torch.autograd import gradcheck

from molgen.modules import SimpleGenerator, SimpleDiscriminator

# Test for SimpleGenerator
def test_SimpleGenerator():
    latent_dim = 100
    output_dim = 784
    hidden_dim = 256
    batch_size = 2
    generator = SimpleGenerator(latent_dim, output_dim, hidden_dim)
    z = torch.randn(batch_size, latent_dim)

    assert generator.forward(z).shape == (batch_size, output_dim)


# Test for SimpleDiscriminator
def test_SimpleDiscriminator():
    output_dim = 784
    hidden_dim = 256
    batch_size = 2
    discriminator = SimpleDiscriminator(output_dim, hidden_dim)
    x = torch.randn(batch_size, output_dim)

    assert discriminator.forward(x).shape == (batch_size, 1)

