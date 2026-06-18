# eo_optimizer.py
from scoring import sum_of_pairs
from column_confidence import compute_column_confidence
from mutations import adaptive_mutation
from reward import compute_reward
import random


def eo_optimize(seed, cfg):
    best = seed
    best_sp = sum_of_pairs(seed)

    for _ in range(cfg.MAX_ITERS):
        col_conf = compute_column_confidence(best)
        cand = adaptive_mutation(best, col_conf, cfg)

        reward, _ = compute_reward(best, cand, cfg)
        cand_sp = sum_of_pairs(cand)

        if reward > 0 or random.random() < 0.05:
            if cand_sp >= best_sp:
                best = cand
                best_sp = cand_sp

    return best, best_sp
