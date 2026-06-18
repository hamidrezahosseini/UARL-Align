# reward.py

from scoring import sum_of_pairs
from uncertainty import alignment_uncertainty


def compute_reward(old_aln, new_aln, cfg, eps=1e-8):
    sp_old = sum_of_pairs(old_aln)
    sp_new = sum_of_pairs(new_aln)

    u_old = alignment_uncertainty(old_aln)
    u_new = alignment_uncertainty(new_aln)

    # relative improvements
    rel_sp = (sp_new - sp_old) / (abs(sp_old) + eps)
    delta_u = u_old - u_new  # positive is good

    reward = (
        cfg.ALPHA * rel_sp
        + cfg.BETA * delta_u
    )

    return reward, {
        "rel_sp": rel_sp,
        "delta_u": delta_u
    }
