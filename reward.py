"""
Fixed reward: now includes the gap-penalty term (γ).
"""

from scoring import sum_of_pairs
from uncertainty import alignment_uncertainty


def compute_reward(old_aln, new_aln, cfg, eps=1e-8):
    sp_old = sum_of_pairs(old_aln)
    sp_new = sum_of_pairs(new_aln)

    u_old = alignment_uncertainty(old_aln, getattr(cfg, "SEQUENCE_TYPE", "auto"))
    u_new = alignment_uncertainty(new_aln, getattr(cfg, "SEQUENCE_TYPE", "auto"))

    g_old = sum(s.count("-") for s in old_aln)
    g_new = sum(s.count("-") for s in new_aln)
    n, L = len(new_aln), len(new_aln[0]) if new_aln else 1

    rel_sp  = (sp_new - sp_old) / (abs(sp_old) + eps)
    delta_u = u_old - u_new                     # positive is good
    delta_g = (g_new - g_old) / max(1, n * L)   # positive = more gaps (bad)

    reward = (
        cfg.ALPHA * rel_sp
        + cfg.BETA  * delta_u
        - cfg.GAMMA * delta_g        # ← now used!
    )

    return reward, {
        "rel_sp":  rel_sp,
        "delta_u": delta_u,
        "delta_g": delta_g,
    }