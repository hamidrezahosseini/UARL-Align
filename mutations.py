# mutations.py

import random
import copy


def select_target_columns(col_unc, threshold):
    return [i for i, u in enumerate(col_unc) if u > threshold]


def targeted_block_shift(aln, target_cols, max_shift, block_range):
    aln = copy.deepcopy(aln)
    if not target_cols:
        return aln

    n = len(aln)
    L = len(aln[0])

    s = random.randint(0, n - 1)
    center = random.choice(target_cols)
    block_len = random.randint(*block_range)

    start = max(0, center - block_len // 2)
    end = min(L, start + block_len)

    row = aln[s]
    block = row[start:end]
    rest = row[:start] + row[end:]

    shift = random.randint(-max_shift, max_shift)
    new_pos = max(0, min(len(rest), start + shift))

    aln[s] = rest[:new_pos] + block + rest[new_pos:]
    return aln


def targeted_column_swap(aln, target_cols, window):
    aln = copy.deepcopy(aln)
    if len(target_cols) < 2:
        return aln

    i, j = random.sample(target_cols, 2)
    if abs(i - j) > window:
        return aln

    for k in range(len(aln)):
        row = list(aln[k])
        row[i], row[j] = row[j], row[i]
        aln[k] = "".join(row)

    return aln


def adaptive_mutation(aln, col_unc, cfg, phase="exploit"):
    """
    phase: 'explore' or 'exploit'
    default = exploit (for EO final refinement)
    """

    target_cols = select_target_columns(col_unc, cfg.UNCERTAINTY_HIGH)

    if phase == "explore":
        max_shift = cfg.EXPLORE_MAX_SHIFT
        block_range = cfg.EXPLORE_BLOCK_RANGE
        window = cfg.EXPLORE_SWAP_WINDOW
    else:
        max_shift = cfg.EXPLOIT_MAX_SHIFT
        block_range = cfg.EXPLOIT_BLOCK_RANGE
        window = cfg.EXPLOIT_SWAP_WINDOW

    if not target_cols:
        return aln

    if random.random() < 0.6:
        return targeted_block_shift(
            aln, target_cols, max_shift, block_range
        )
    else:
        return targeted_column_swap(
            aln, target_cols, window
        )
