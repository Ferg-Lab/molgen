from torch import nn


class SimpleGenerator(nn.Module):
    """
    SimpleGenerator is a simple implementation of generator module in GANs.
    It takes a latent dimension and an output dimension as input, and has a hidden dimension (default = 256)
    It is implemented as a sequential model with 3 linear layers, batch normalization and SiLU activation functions.
    It is a sub-class of nn.Module.
    
    Parameters
    ----------
    latent_dim : int
        dimension of the latent space

    output_dim : int
        dimension of the output space

    hidden_dim : int, default=256
        dimension of the hidden layers
    
    """
    def __init__(self, latent_dim, output_dim, hidden_dim=256):
        super(SimpleGenerator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.SiLU())
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class SimpleDiscriminator(nn.Module):
    """
    SimpleDiscriminator is a simple implementation of discriminator module in GANs.
    It takes an output dimension as input, and has a hidden dimension (default = 256)
    It is implemented as a sequential model with 3 linear layers and SiLU activation functions.
    It is a sub-class of nn.Module.
    
    Parameters
    ----------
    output_dim : int
        dimension of the output space

    hidden_dim : int, default=256
        dimension of the hidden layers
    
    """
    def __init__(self, output_dim, hidden_dim=256):
        super(SimpleDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity
