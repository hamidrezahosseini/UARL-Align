# pg_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim

from column_confidence import compute_column_confidence
from mutations import adaptive_mutation
from reward import compute_reward


class PGTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode = 0

        self.policy = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softplus()
        ).float()

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=cfg.RL_LR
        )

        self.history = []

    def get_phase(self):
        split = int(self.cfg.N_EPISODES * self.cfg.PHASE_SPLIT)
        return "explore" if self.episode < split else "exploit"

    def train_step(self, aln):
        phase = self.get_phase()

        col_unc = compute_column_confidence(aln)
        u_mean = torch.tensor(
            [[float(col_unc.mean())]],
            dtype=torch.float32
        )

        params = self.policy(u_mean)
        alpha, beta, gamma = params[0]

        # phase-aware weighting
        self.cfg.ALPHA = alpha.item()
        self.cfg.BETA  = beta.item()
        self.cfg.GAMMA = gamma.item() if phase == "exploit" else 0.0

        new_aln = adaptive_mutation(aln, col_unc, self.cfg, phase)

        reward, _ = compute_reward(aln, new_aln, self.cfg)
        reward_t = torch.tensor(reward, dtype=torch.float32)

        loss = -reward_t * params.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.history.append({
            "episode": self.episode,
            "phase": phase,
            "reward": reward,
            "alpha": self.cfg.ALPHA,
            "beta": self.cfg.BETA,
            "gamma": self.cfg.GAMMA
        })

        self.episode += 1
        return self.history[-1]
