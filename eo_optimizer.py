
import random

from scoring import sum_of_pairs
from uncertainty import column_uncertainties, alignment_uncertainty
from mutations import adaptive_mutation
from reward import compute_reward


def _validate_candidate(original, candidate):
    """Check alignment dimensions and preservation of each sequence."""
    if len(original) != len(candidate):
        raise ValueError("Mutation changed the number of sequences.")

    if not original:
        return

    original_widths = {len(seq) for seq in original}
    candidate_widths = {len(seq) for seq in candidate}

    if len(original_widths) != 1 or len(candidate_widths) != 1:
        raise ValueError("Alignment rows must have equal lengths.")

    if original_widths != candidate_widths:
        raise ValueError("Mutation changed the alignment width.")

    for old_seq, new_seq in zip(original, candidate):
        if old_seq.replace("-", "") != new_seq.replace("-", ""):
            raise ValueError(
                "Mutation changed the ungapped sequence or residue order."
            )


def eo_optimize(seed, cfg):
    best = list(seed)
    best_sp = sum_of_pairs(best)

    sequence_type = getattr(cfg, "SEQUENCE_TYPE", "auto")

    for _ in range(cfg.MAX_ITERS):
        # Compute uncertainty directly; do not pass confidence values here.
        col_uncertainty = column_uncertainties(
            best,
            sequence_type=sequence_type,
        )

        cand = adaptive_mutation(
            best,
            col_uncertainty,
            cfg,
            phase="exploit",
        )

        _validate_candidate(best, cand)

        # Skip unchanged candidates to avoid unnecessary scoring.
        if cand == best:
            continue

        reward, _ = compute_reward(best, cand, cfg)
        cand_sp = sum_of_pairs(cand)

        # Retain the existing acceptance policy.
        if (reward > 0 or random.random() < 0.05) and cand_sp >= best_sp:
            best = cand
            best_sp = cand_sp

    return best, best_sp
