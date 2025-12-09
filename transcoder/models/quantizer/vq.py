from einops import rearrange
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple
import math
from torchlambertw import special


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        with torch.amp.autocast("cuda", enabled=False):
            self.parameters = parameters
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
            self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
            self.deterministic = deterministic
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)
            if self.deterministic:
                self.var = self.std = torch.zeros_like(self.mean).to(
                    device=self.parameters.device
                )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, kl_mode, group, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                MODE = kl_mode
                FLIP_FACTOR = 40
                TOLERANCE = 0.35
                # 2.77 * 4 -> 16
                # for now use 12  is perhaps better? ...
                TARGET = 2.77 * 4
                GROUP = group
                if MODE == "even":
                    return 0.5 * torch.sum(
                        torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                        dim=[1, 2, 3],
                    )
                elif MODE == "target":
                    kl = 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
                    b, c, h, w = kl.shape
                    kl = kl.reshape(b, GROUP, c // GROUP, h, w)
                    kl = torch.sum(kl, dim=1)
                    dtype = kl.dtype
                    ge = (kl > TARGET + TOLERANCE).type(dtype) * FLIP_FACTOR
                    eq = (kl <= TARGET + TOLERANCE).type(dtype) * (
                        kl >= TARGET - TOLERANCE
                    ).type(dtype)
                    le = (kl < TARGET - TOLERANCE).type(dtype) * (1 / FLIP_FACTOR)
                    kl_sum = torch.sum(ge * kl + eq * kl + le * kl, dim=[1, 2, 3])
                    return kl_sum
                elif MODE == "mean":
                    kl = 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
                    dtype = kl.dtype
                    kl_mean = torch.mean(kl.detach())
                    ge = (kl > kl_mean + TOLERANCE).type(dtype) * FLIP_FACTOR
                    eq = (kl <= kl_mean + TOLERANCE).type(dtype) * (
                        kl >= kl_mean - TOLERANCE
                    ).type(dtype)
                    le = (kl < kl_mean - TOLERANCE).type(dtype) * (1 / FLIP_FACTOR)

                    kl_sum = torch.sum(ge * kl + eq * kl + le * kl, dim=[1, 2, 3])
                    return kl_sum
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class DiagonalGaussianRegularizer(nn.Module):
    def __init__(self, kl_mode, group, input_format):
        super().__init__()
        self.sample = True
        self.kl_mode = kl_mode
        assert input_format in ["bchw", "blc"]
        self.input_format = input_format
        self.group = group

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z = z.float()
        if self.input_format == "blc":
            z = rearrange(z, "b l c -> b c l")
            b, c, l = z.shape
            h = int(math.sqrt(l))
            z = z.reshape(b, c, h, h)

        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()

        kl_loss = posterior.kl(self.kl_mode, self.group)
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        log["kl_loss"] = kl_loss.detach()
        log["mean"] = posterior.mean.detach()
        log["logvar"] = posterior.logvar.detach()
        log["std"] = posterior.std.detach()
        log["var"] = posterior.var.detach()

        kls = 1.4426 * (
            0.5
            * (torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar)
        )

        GROUP = self.group
        b, c, h, w = kls.shape
        klsg = torch.sum(kls.reshape(b, GROUP, c // GROUP, h, w), dim=1).detach()

        log["H"] = torch.mean(klsg)
        log["H-max"] = torch.max(klsg)
        log["H-min"] = torch.min(klsg)

        if self.input_format == "blc":
            z = z.reshape(b, c, -1)
            z = rearrange(z, "b l c -> b c l")

        return z, kl_loss, log


class TargetAdaptativeRegularizer(nn.Module):
    def __init__(self, target, group, input_format):
        super().__init__()
        self.sample = True
        self.target = target
        assert input_format in ["bchw", "blc"]
        self.input_format = input_format
        self.group = group
        self.target = target
        self.lam_factor = 1 + 1e-2
        self.lam = 1.0
        self.lam_min = 1.0
        self.lam_max = 1.0
        self.lam_range = (1e-3, 1e3)
        self.tolerance = 0.5

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z = z.float()
        if self.input_format == "blc":
            z = rearrange(z, "b l c -> b c l")
            b, c, l = z.shape
            h = int(math.sqrt(l))
            z = z.reshape(b, c, h, h)

        # internally bchw
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()

        kls = 1.4426 * (
            0.5
            * (torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar)
        )

        b, c, h, w = kls.shape
        kls = torch.sum(kls.reshape(b, self.group, c // self.group, h, w), dim=1)

        ge = (kls > self.target + self.tolerance).type(kls.dtype) * self.lam_max
        eq = (kls <= self.target + self.tolerance).type(kls.dtype) * (
            kls >= self.target - self.tolerance
        ).type(kls.dtype)

        le = (kls < self.target - self.tolerance).type(kls.dtype) * self.lam_min
        kl_loss = torch.sum((ge * kls + eq * kls + le * kls), dim=[1,2,3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = kl_loss * self.lam

        log["kl_loss"] = kl_loss.detach()
        log["mean"] = posterior.mean.detach()
        log["logvar"] = posterior.logvar.detach()
        log["std"] = posterior.std.detach()
        log["var"] = posterior.var.detach()

        log["H"] = torch.mean(kls).detach()
        log["H-max"] = torch.max(kls).detach()
        log["H-min"] = torch.min(kls).detach()

        if torch.mean(kls) > self.target:
            self.lam = self.lam * self.lam_factor
        else:
            self.lam = self.lam / self.lam_factor

        if torch.max(kls) > self.target + self.tolerance:
            self.lam_max = self.lam_max * self.lam_factor
        else:
            self.lam_max / self.lam_max * self.lam_factor
        self.lam_max = max(min(self.lam_max, self.lam_range[1]), 1.0)

        if torch.min(kls) < self.target - self.tolerance:
            self.lam_min = self.lam_min / self.lam_factor
        else:
            self.lam_min = self.lam_min * self.lam_factor
        self.lam_min = max(min(self.lam_min, 1.0), self.lam_range[0])

        if self.input_format == "blc":
            z = z.reshape(b, c, -1)
            z = rearrange(z, "b l c -> b c l")

        return z, kl_loss, log

class GroupedLambertWRegularizer(nn.Module):
    # import torch
    # from sgm.modules.autoencoding.regularizers import GroupedLambertWRegularizer
    # lambert = GroupedLambertWRegularizer(11.091, 16)
    # z, out = lambert(torch.randn([2,32,8,8]))
    # print(out["bits-mean"])
    def __init__(self, kl, group, input_format):
        super().__init__()
        self.kl = kl
        self.group = group
        self.input_format = input_format

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        # force fp32
        z = z.float()

        if self.input_format == "blc":
            z = rearrange(z, "b l c -> b c l")
            zb, zc, zl = z.shape
            zh = int(math.sqrt(zl))
            z = z.reshape(zb, zc, zh, zh)

        tau, gamma_unnorm = torch.chunk(z, 2, dim=1)

        b, c, h, w = tau.shape
        assert(c % self.group == 0)
        ng = c // self.group

        tau = tau.reshape(b, self.group, ng, h, w)
        gamma_unnorm = gamma_unnorm.reshape(b, self.group, ng, h, w)

        gamma = F.softmax(gamma_unnorm, dim=1)

        # shrink upperbound of mu a little to avoid nan in lambert
        kw = gamma * (self.kl - 0.1) + 0.1 / self.group
    
        mu = torch.sqrt(2 * kw) * F.tanh(tau) * (1 - 1e-2)

        W = special.lambertw

        var = -W(-torch.exp(mu ** 2 - 2 * kw - 1.0))

        if torch.isnan(var).any():
            var = var.reshape(-1)
            mu = mu.reshape(-1)
            kw = kw.reshape(-1)
            for i in range(var.shape[0]):
                if torch.isnan(var[i]):
                    print("caught nan")
                    print(mu[i], kw[i])
                    print(mu[i].dtype, kw[i].dtype)
            print("var nan")
            assert(0)

        std = torch.sqrt(var)
        logvar = torch.log(var)

        kls = (
            1.4426 * (0.5 * (torch.pow(mu, 2) + var - 1.0 - logvar)).clone().detach()
        )
        kls = torch.sum(kls, dim=1)

        log["H"] = torch.mean(kls)
        log["H-max"] = torch.max(kls)
        log["H-min"] = torch.min(kls)

        sample = mu + std * torch.randn_like(mu)

        sample = sample.reshape(b, c, h, w)
        kl_loss = torch.tensor(0.0).to(device=sample.device)
        log["kl_loss"] = kl_loss
        log["mean"] = mu.reshape(b, c, h, w).clone().detach()
        log["logvar"] = logvar.reshape(b, c, h, w).clone().detach()
        log["std"] = std.reshape(b, c, h, w).clone().detach()
        log["var"] = var.reshape(b, c, h, w).clone().detach()

        if self.input_format == "blc":
            sample = sample.reshape(b, c, -1)
            sample = rearrange(sample, "b l c -> b c l")

        return sample, kl_loss, log


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, l2_norm, beta, input_format="bchw"):
        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        assert input_format in ["bchw", "blc"]
        self.input_format = input_format

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1 / n_embed, 1 / n_embed)
        self.bits_per_index = int(np.ceil(np.log2(n_embed)))

    def forward(self, z):
        batch = z.shape[0]
        if self.input_format == "bchw":
            z = rearrange(z, "b c h w -> b h w c")

        if self.l2_norm:
            z = F.normalize(z, dim=-1)
            z_flatten = z.reshape(-1, self.embed_dim)
            embedding_weight = F.normalize(self.embedding.weight, dim=-1)
            d = -z_flatten @ embedding_weight.t()
        else:
            z_flatten = z.reshape(-1, self.embed_dim)
            d = (
                torch.sum(z_flatten**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * z_flatten @ self.embedding.weight.t()
            )

        min_encoding_indices = torch.argmin(d.detach(), dim=1)
        if not self.training:
            used_codes = torch.unique(min_encoding_indices, return_counts=False)
        else:
            used_codes = None
        cb_usage = F.one_hot(min_encoding_indices, self.n_embed).sum(0)
        cb_entropy = self.get_entropy(cb_usage)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        # fix the issue with loss scaling
        # loss weight should not associate with the dimensionality of words
        # loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        loss = self.beta * torch.mean(
            ((z_q.detach() - z) ** 2).sum(dim=-1)
        ) + torch.mean(((z_q - z.detach()) ** 2).sum(dim=-1))

        z_q = z + (z_q - z).detach()
        if self.input_format == "bchw":
            z_q = rearrange(z_q, "b h w c -> b c h w")
        return (
            z_q,
            loss,
            {
                "H": cb_entropy,
                "H-min": cb_entropy,
                "H-max": cb_entropy,
                "used_codes": used_codes,
                "indices": min_encoding_indices.view(batch, -1),
            },
        )

    def get_entropy(self, count, eps=1e-4):
        probs = (count + eps) / (count + eps).sum()
        H = -(probs * torch.log(probs)).sum()
        return H

    def get_codebook_entry(self, indices):
        z_q = self.embedding(indices)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        if self.input_format == "bchw":
            h = w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], "Invalid sequence length"
            z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h)
        return z_q


class FSQQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, l2_norm, beta, input_format="bchw"):
        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        assert input_format in ["bchw", "blc"]
        self.input_format = input_format

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1 / n_embed, 1 / n_embed)
        self.bits_per_index = int(np.ceil(np.log2(n_embed)))

    def forward(self, z):
        batch = z.shape[0]
        if self.input_format == "bchw":
            z = rearrange(z, "b c h w -> b h w c")

        if self.l2_norm:
            z = F.normalize(z, dim=-1)
            z_flatten = z.reshape(-1, self.embed_dim)
            embedding_weight = F.normalize(self.embedding.weight, dim=-1)
            d = -z_flatten @ embedding_weight.t()
        else:
            z_flatten = z.reshape(-1, self.embed_dim)
            d = (
                torch.sum(z_flatten**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * z_flatten @ self.embedding.weight.t()
            )

        min_encoding_indices = torch.argmin(d.detach(), dim=1)
        if not self.training:
            used_codes = torch.unique(min_encoding_indices, return_counts=False)
        else:
            used_codes = None
        cb_usage = F.one_hot(min_encoding_indices, self.n_embed).sum(0)
        cb_entropy = self.get_entropy(cb_usage)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        # fix the issue with loss scaling
        # loss weight should not associate with the dimensionality of words
        # loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        loss = self.beta * torch.mean(
            ((z_q.detach() - z) ** 2).sum(dim=-1)
        ) + torch.mean(((z_q - z.detach()) ** 2).sum(dim=-1))

        z_q = z + (z_q - z).detach()
        if self.input_format == "bchw":
            z_q = rearrange(z_q, "b h w c -> b c h w")
        return (
            z_q,
            loss,
            {
                "H": cb_entropy,
                "H-min": cb_entropy,
                "H-max": cb_entropy,
                "used_codes": used_codes,
                "indices": min_encoding_indices.view(batch, -1),
            },
        )

    def get_entropy(self, count, eps=1e-4):
        probs = (count + eps) / (count + eps).sum()
        H = -(probs * torch.log(probs)).sum()
        return H

    def get_codebook_entry(self, indices):
        z_q = self.embedding(indices)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        if self.input_format == "bchw":
            h = w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], "Invalid sequence length"
            z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h)
        return z_q


class EMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embed,
        embed_dim,
        l2_norm,
        beta,
        decay=0.99,
        eps=1e-5,
        random_restart=True,
        restart_threshold=1.0,
        input_format="bchw",
    ):
        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.random_restart = random_restart
        self.restart_threshold = restart_threshold
        self.input_format = input_format

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(
            -1 / n_embed, 1 / n_embed
        )  # TODO (yzhao): test other initialization methods
        self.register_buffer("ema_cluster_size", torch.zeros(self.n_embed))
        self.embedding_avg = nn.Parameter(torch.Tensor(self.n_embed, self.embed_dim))
        self.embedding_avg.data.copy_(self.embedding.weight.data)

    def _tile(self, z):
        n_z, embedding_dim = z.shape
        if n_z < self.n_embed:
            n_repeats = (self.n_embed + n_z - 1) // n_z
            std = 0.01 / np.sqrt(embedding_dim)
            z = z.repeat(n_repeats, 1)
            z = z + torch.randn_like(z) * std
        return z

    def forward(self, z):
        if self.input_format == "bchw":
            z = rearrange(z, "b c h w -> b h w c")
        z_flatten = z.reshape(-1, self.embed_dim)

        d = (
            torch.sum(z_flatten**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * z_flatten @ self.embedding.weight.t()
        )

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.n_embed, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        if self.l2_norm:
            z = F.normalize(z, dim=-1)
            z_q = F.normalize(z_q, dim=-1)

        if self.training:
            # EMA update cluster size
            encodings_sum = encodings.sum(0)
            if dist.is_initialized():
                dist.all_reduce(encodings_sum)
            self.ema_cluster_size.data.mul_(self.decay).add_(
                encodings_sum, alpha=1 - self.decay
            )

            # EMA update of the embedding vectors
            dw = encodings.t() @ z_flatten
            if dist.is_initialized():
                dist.all_reduce(dw)
            self.embedding_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size)
            weights = (
                (self.ema_cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            self.embedding.weight.data = self.embedding_avg.data / weights.unsqueeze(1)

            if self.random_restart:
                zz = self._tile(z_flatten)
                _k_rand = zz[torch.randperm(zz.size(0))][: self.n_embed]
                if dist.is_initialized():
                    dist.broadcast(_k_rand, 0)
                usage = (
                    self.ema_cluster_size.view(-1, 1) > self.restart_threshold
                ).float()
                self.embedding.weight.data.mul_(usage).add_(_k_rand * (1 - usage))

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        z_q = z + (z_q - z).detach()
        if self.input_format == "bchw":
            z_q = rearrange(z_q, "b h w c -> b c h w")
        # TODO (yzhao): monitor utility of the dictionary
        return z_q, loss, {}
