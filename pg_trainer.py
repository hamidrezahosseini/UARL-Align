"""
Proper REINFORCE policy-gradient for alignment refinement.

Key fixes:
    1. Gaussian policy over weights (continuous action space).
    2. Action is SAMPLED (not deterministic) → exploration.
    3. Training is INTERLEAVED with search (diverse states).
    4. Standard policy-gradient loss with return-to-go and EMA baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PGTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode = 0
        self.baseline = 0.0

        # 3-D state: [mean U, std U, gap_ratio] → 3-D Gaussian mean
        self.policy = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        ).float()
        self.log_std = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0]))
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + [self.log_std],
            lr=cfg.RL_LR,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _state(aln):
        from uncertainty import column_uncertainties
        try:
            u = column_uncertainties(aln)
        except Exception:
            u = [0.5] * len(aln[0])
        u_mean = float(np.mean(u)) if u else 0.0
        u_std = float(np.std(u)) if u else 0.0
        n, L = len(aln), len(aln[0])
        g = sum(s.count("-") for s in aln)
        g_ratio = g / (n * L) if n * L else 0.0
        return np.array([u_mean, u_std, g_ratio], dtype=np.float32)

    # ------------------------------------------------------------------
    def sample_weights(self, aln):
        """Sample weights from N(μ(s), σ²) and return (α, β, γ, log_prob)."""
        s = torch.from_numpy(self._state(aln))
        mean = self.policy(s)
        std = torch.exp(self.log_std).clamp(0.1, 1.0)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()                 # reparametrized → gradients flow
        w = torch.sigmoid(z)               # bounded to (0, 1)
        a, b, c = float(w[0]), float(w[1]), float(w[2])
        # Log-probability of sigmoid-transformed Gaussian (Jacobian)
        jac = torch.log(w * (1 - w) + 1e-8).sum()
        log_prob = dist.log_prob(z).sum() - jac
        return a, b, c, log_prob

    # ------------------------------------------------------------------
    def update(self, log_probs, rewards, gamma=0.99):
        """REINFORCE with return-to-go and EMA baseline."""
        if not log_probs:
            return
        returns = np.zeros(len(rewards), dtype=np.float64)
        R = 0.0
        for k in range(len(rewards) - 1, -1, -1):
            R = rewards[k] + gamma * R
            returns[k] = R

        advantages = torch.tensor(returns - self.baseline, dtype=torch.float32)

        loss = 0.0
        for lp, adv in zip(log_probs, advantages):
            loss = loss - lp * adv

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + [self.log_std], 1.0
        )
        self.optimizer.step()

        for r in returns:
            self.baseline = 0.95 * self.baseline + 0.05 * r