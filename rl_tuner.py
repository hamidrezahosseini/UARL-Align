# rl_tuner.py

import random
from eo_optimizer import eo_optimize
from scoring import sum_of_pairs
from uncertainty import alignment_uncertainty

class RLTuner:
    def __init__(self, cfg):
        # initial parameters
        self.alpha = cfg.ALPHA
        self.beta  = cfg.BETA
        self.gamma = cfg.GAMMA

        self.cfg = cfg

        self.actions = [
            (0.1, 0, 0),
            (-0.1, 0, 0),
            (0, 0.1, 0),
            (0, -0.1, 0),
            (0, 0, 0.1),
            (0, 0, -0.1)
        ]

    def step(self, seed_alignment):
        base_sp = sum_of_pairs(seed_alignment)
        base_unc = alignment_uncertainty(seed_alignment)

        da, db, dg = random.choice(self.actions)

        # candidate parameters
        new_alpha = max(0.1, self.alpha + da)
        new_beta  = max(0.0, self.beta  + db)
        new_gamma = max(0.0, self.gamma + dg)

        # temporarily update cfg
        self.cfg.ALPHA = new_alpha
        self.cfg.BETA  = new_beta
        self.cfg.GAMMA = new_gamma

        best_aln, _ = eo_optimize(seed_alignment, self.cfg)

        new_sp = sum_of_pairs(best_aln)
        new_unc = alignment_uncertainty(best_aln)

        reward = (new_sp - base_sp) - 0.5 * (new_unc - base_unc)

        # accept or reject
        if reward > 0:
            self.alpha, self.beta, self.gamma = new_alpha, new_beta, new_gamma
        else:
            # rollback
            self.cfg.ALPHA = self.alpha
            self.cfg.BETA  = self.beta
            self.cfg.GAMMA = self.gamma

        return reward
