import torch
from torch import nn
from einops import rearrange
from molgen.utils import (
    SinusoidalPosEmb,
    Mish,
    Residual,
    default,
    exists,
    linear_schedule,
    cosine_beta_schedule,
    extract,
    generate_inprint_mask,
    noise_like,
)
import numpy as np
from tqdm.autonotebook import tqdm
from functools import partial


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


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        # b, c, l = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) l -> qkv b heads c l", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c l -> b (heads c) l", heads=self.heads)
        return self.to_out(out)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Unet(nn.Module):
    """From: https://github.com/lucidrains/denoising-diffusion-pytorch"""

    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), groups=8):
        super().__init__()
        dims = [1, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.feature_dim = dim
        self.dim_mults = dim_mults
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out * 2, dim_in, time_emb_dim=dim, groups=groups
                        ),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, 1)
        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups), nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        size_list = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            size_list.append(x.shape[-1])
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:

            x = torch.cat((x[:, :, : size_list.pop()], h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x[:, :, : size_list.pop()])


class GaussianDiffusion(nn.Module):
    """From: https://github.com/lucidrains/denoising-diffusion-pytorch"""

    def __init__(
        self,
        denoise_fn,
        timesteps=1000,
        loss_type="l1",
        betas=None,
        beta_schedule="linear",
        unmask_number=0,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )

        # which beta scheduler to use
        else:
            if beta_schedule == "linear":
                betas = linear_schedule(timesteps)
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.unmask_number = unmask_number
        if unmask_number == 0:
            self.unmask_index = None
        else:
            self.unmask_index = [*range(unmask_number)]

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, _, l, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        denosied_x = (
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        )
        inprint_mask = generate_inprint_mask(b, l, self.unmask_index).to(device)

        denosied_x[inprint_mask] = x[inprint_mask]

        return denosied_x

    @torch.no_grad()
    def p_sample_loop(self, shape, samples=None):
        device = self.betas.device

        b = shape[0]
        state = torch.randn(shape, device=device)

        # if not samples == None:
        if samples is not None:
            assert shape == samples.shape

            inprint_mask = generate_inprint_mask(b, shape[2], self.unmask_index).to(
                device
            )
            state[inprint_mask] = samples[inprint_mask]

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            state = self.p_sample(
                state, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return state

    @torch.no_grad()
    def sample(self, op_number, batch_size=16, samples=None):
        return self.p_sample_loop((batch_size, 1, op_number), samples)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        # if not self.unmask_index == None:
        if self.unmask_index is not None:
            b, c, l = x_start.shape
            inprint_mask = generate_inprint_mask(b, l, self.unmask_index).to(
                x_start.device
            )
            x_start[inprint_mask]
            x_noisy[inprint_mask] = x_start[inprint_mask]
        else:
            inprint_mask = None
        return x_noisy, inprint_mask

    def p_losses(self, x_start, t, noise=None):
        b, c, l = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy, inprint_mask = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(x_noisy, t)

        # if not inprint_mask == None:
        if inprint_mask is not None:
            noise = torch.masked_select(noise, ~inprint_mask)
            x_recon = torch.masked_select(x_recon, ~inprint_mask)

        if self.loss_type == "l1":
            loss = torch.nn.functional.l1_loss(noise, x_recon)
        elif self.loss_type == "l2":
            loss = torch.nn.functional.mse_loss(noise, x_recon)
        elif self.loss_type == "huber":
            loss = torch.nn.functional.smooth_l1_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)
