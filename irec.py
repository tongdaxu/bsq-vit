import torch
import torch.nn as nn
from torch.distributions import Normal, Exponential
from torch.distributions.gumbel import Gumbel
from torch.quasirandom import SobolEngine

from scipy.stats import norm
import math


def cdf(x):
    return 0.5 * (1 + torch.erf((x) / math.sqrt(2)))


class GaussianPTQ(nn.Module):
    def __init__(self, n):
        super().__init__()
        # divide Gaussian distribution into n parts with same probability
        self.normal = torch.distributions.normal.Normal(
            torch.zeros([1]), torch.ones([1])
        )
        bins = []
        for i in range(1, n):
            bins.append(self.normal.icdf(torch.tensor(i / n)).item())
        centers = [-3.0]
        for i in range(len(bins) - 1):
            centers.append((bins[i] + bins[i + 1]) / 2)
        centers.append(3.0)
        self.centers = nn.Parameter(torch.tensor(centers)[None])

    def forward(self):
        raise NotImplementedError

    def encode(self, sample):
        index = torch.argmin(torch.abs(self.centers - sample), dim=1)
        received_samples = torch.index_select(self.centers, 1, index).reshape(-1, 1)
        return received_samples


class GaussianPFR(nn.Module):
    def __init__(self, n_samples, dim, seed):
        super().__init__()
        self.n_samples = n_samples
        self.prior_samples = nn.Parameter(
            prior_samples(self.n_samples, dim, seed).float()
        )
        self.normal_dist = Normal(torch.zeros([1, dim]), torch.ones([1, dim]))
        self.normal_log_prob = nn.Parameter(
            self.normal_dist.log_prob(self.prior_samples)
        )
        self.zero = nn.Parameter(torch.tensor(0.0))
        self.one = nn.Parameter(torch.tensor(1.0))
        self.seed = seed

    def forward(self):
        raise NotImplementedError

    def encode(self, mu_q, std_q, mode):
        # pfr, only accepts 1d tensor in batch term
        assert len(mu_q.shape) == 2 and len(std_q.shape) == 2
        assert mu_q.shape[0] == std_q.shape[0]

        bs = mu_q.shape[0]
        q_normal_dist = Normal(mu_q[:, None, :], std_q[:, None, :])

        # ORC, but does not make a big difference?
        # need more test?
        if mode == "orc":
            # exp_dist = Exponential(self.one)
            # exps = exp_dist.sample((bs, self.n_samples))
            gumbel_dist = Gumbel(self.zero, self.one)
            perturbs = gumbel_dist.sample((bs, self.n_samples))
            perturbs, _ = torch.sort(perturbs, dim=1, descending=False)
            log_ratios = (
                q_normal_dist.log_prob(self.prior_samples[None])
                - self.normal_log_prob[None]
            )
            perturbed = torch.sum(log_ratios, dim=2) + perturbs
        elif mode == "mrc":
            gumbel_dist = Gumbel(self.zero, self.one)
            perturbs = gumbel_dist.sample((bs, self.n_samples))
            log_ratios = (
                q_normal_dist.log_prob(self.prior_samples[None])
                - self.normal_log_prob[None]
            )
            perturbed = torch.sum(log_ratios, dim=2) + perturbs
        elif mode == "det":
            log_ratios = (
                q_normal_dist.log_prob(self.prior_samples[None])
                - self.normal_log_prob[None]
            )
            perturbed = torch.sum(log_ratios, dim=2)
        elif mode == "mle":
            log_ratios = q_normal_dist.log_prob(self.prior_samples[None])
            perturbed = torch.sum(log_ratios, dim=2)
        else:
            raise ValueError
        # training ...
        # gumbel_one_hot = torch.nn.functional.gumbel_softmax(perturbed, tau=0.1, hard=True, eps=1e-10, dim=1)
        # received_samples = torch.sum(self.prior_samples[None] * gumbel_one_hot[:,:,None], dim=1)

        argmax_indices = torch.argmax(perturbed, dim=1)
        received_samples = torch.index_select(self.prior_samples, 0, argmax_indices)

        return received_samples, argmax_indices

    def decode(self, argmax_indices):
        return torch.index_select(self.prior_samples, 0, argmax_indices)


def prior_samples(n_samples, n_variable, seed_rec):
    sobol = SobolEngine(n_variable, scramble=True, seed=seed_rec)
    samples_sobol = sobol.draw(n_samples)
    samples_i = torch.from_numpy(norm.ppf(samples_sobol))
    # samples_i = torch.clamp(samples_i, -10, 10)
    return samples_i


def iREC(kl2_budget, seed_rec, mu_q, std_q, mu_p, std_p, standard_samples):
    # fixed bit budget
    device = mu_q.device
    n_samples = 2**kl2_budget
    n_variable = mu_p.shape[0]

    N = 1
    exp_dist = Exponential(torch.tensor(1.0).to(device=device))
    deltas = exp_dist.sample((N, n_samples))
    ts = torch.cumsum(deltas, dim=1)
    gumbels = -torch.log(ts)
    normal_dist = Normal(
        mu_p * torch.ones(N, n_samples, n_variable).to(device=device),
        std_p * torch.ones(N, n_samples, n_variable).to(device=device),
    )

    xs = standard_samples * std_p + mu_p
    xs = xs.unsqueeze(0)

    q_normal_dist = Normal(mu_q, std_q)
    log_ratios = q_normal_dist.log_prob(xs) - normal_dist.log_prob(xs)
    log_ratios = log_ratios.sum(dim=-1)

    # importance weight perturbed with ordered Gumbel noise
    perturbed = log_ratios + gumbels

    argmax_indices = torch.argmax(perturbed, dim=1)
    approx_samp_mask = torch.zeros(
        N, n_samples, dtype=torch.bool, device=argmax_indices.device
    )
    approx_samp_mask.scatter_(1, argmax_indices.unsqueeze(1), True)

    approx_samps = xs[approx_samp_mask]
    received_samples = approx_samps
    return received_samples[0], argmax_indices.item()


if __name__ == "__main__":
    klbound = 22
    seed = 64
    samples, index = iREC(
        klbound,
        seed,
        torch.tensor([3.0, 3.0]),
        torch.tensor([1e-3, 1e-2]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0]),
    )
    print(samples, index)
