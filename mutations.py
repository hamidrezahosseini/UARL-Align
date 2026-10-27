"""
Stronger, uncertainty-aware mutation operators.

Fixes over the original:
  • Fallback mechanism when no valid target column exists.
  • Multiple operators: gap relocation, gap shift, column swap.
  • `force=True` guarantees a change when possible.
"""

import random
import numpy as np


def select_target_columns(col_uncertainty, threshold, mode="high"):
    """Select high- or low-uncertainty columns."""
    if mode == "high":
        return [i for i, u in enumerate(col_uncertainty) if u > threshold]
    return [i for i, u in enumerate(col_uncertainty) if u < threshold]


# ---------------------------------------------------------------------------
# Operator 1: relocate a gap toward a target column (existing, improved)
# ---------------------------------------------------------------------------
def move_gap_toward_target(aln, target_cols):
    if not aln or not target_cols:
        return [seq[:] for seq in aln]

    n_cols = len(aln[0])

    # Collect all valid (row, target, gap_positions) triples.
    eligible = []
    for row_idx, seq in enumerate(aln):
        gap_positions = [i for i, c in enumerate(seq) if c == "-"]
        if not gap_positions:
            continue
        for target in target_cols:
            if 0 <= target < n_cols and seq[target] != "-":
                eligible.append((row_idx, target, gap_positions))

    if not eligible:
        return [seq[:] for seq in aln]

    row_idx, target, gap_positions = random.choice(eligible)
    gap_idx = random.choice(gap_positions)
    if gap_idx == target:
        return [seq[:] for seq in aln]

    row = list(aln[row_idx])
    row.pop(gap_idx)
    row.insert(target, "-")
    candidate = [seq[:] for seq in aln]
    candidate[row_idx] = "".join(row)

    if candidate[row_idx].replace("-", "") != aln[row_idx].replace("-", ""):
        return [seq[:] for seq in aln]
    return candidate


# ---------------------------------------------------------------------------
# Operator 2: block-gap shift (move a gap segment by δ)
# ---------------------------------------------------------------------------
def block_gap_shift(aln, target_cols, max_shift=6):
    if not aln or not target_cols:
        return [seq[:] for seq in aln]
    n_cols = len(aln[0])
    row_idx = random.randrange(len(aln))
    seq = aln[row_idx]
    gaps = [i for i, c in enumerate(seq) if c == "-"]
    if not gaps:
        return [seq[:] for seq in aln]
    gap_idx = random.choice(gaps)
    delta = random.randint(-max_shift, max_shift)
    if delta == 0:
        delta = 1
    new_pos = max(0, min(n_cols - 1, gap_idx + delta))
    if new_pos == gap_idx:
        return [seq[:] for seq in aln]
    row = list(seq)
    row.pop(gap_idx)
    row.insert(new_pos, "-")
    candidate = [s[:] for s in aln]
    candidate[row_idx] = "".join(row)
    if candidate[row_idx].replace("-", "") != seq.replace("-", ""):
        return [s[:] for s in aln]
    return candidate


# ---------------------------------------------------------------------------
# Operator 3: column swap (exchange two adjacent columns)
# ---------------------------------------------------------------------------
def column_swap(aln, target_cols):
    if not aln or not target_cols or len(aln[0]) < 2:
        return [s[:] for s in aln]
    n_cols = len(aln[0])
    j = random.choice(target_cols)
    # pick a neighbor column to swap with
    k = j + random.choice([-1, 1])
    if not (0 <= k < n_cols):
        return [s[:] for s in aln]
    return [s[:j] + s[k] + s[j + 1:k] + s[j] + s[k + 1:] for s in aln]


# ---------------------------------------------------------------------------
# Meta-operator: try operators until one succeeds
# ---------------------------------------------------------------------------
def adaptive_mutation(aln, col_uncertainty, cfg, phase="exploit", force=False):
    """
    Apply a single mutation, trying multiple operators if needed.

    When `force=True`, retries with random targets until a change is made
    (or exhausts attempts, returning the original).
    """
    threshold = cfg.UNCERTAINTY_HIGH

    # High-U columns are the primary targets (paper's intent).
    targets_high = select_target_columns(col_uncertainty, threshold, "high")

    # In explore phase also allow low-U columns for diversity.
    if phase == "explore" or force:
        targets_low = select_target_columns(col_uncertainty,
                                             cfg.UNCERTAINTY_LOW, "low")
    else:
        targets_low = []

    # Fallback: if no high-U target, use all columns
    if not targets_high:
        targets_high = list(range(len(aln[0])))

    max_shift = (cfg.EXPLORE_MAX_SHIFT if phase == "explore"
                 else cfg.EXPLOIT_MAX_SHIFT)

    operators = [
        lambda: move_gap_toward_target(aln, targets_high),
        lambda: block_gap_shift(aln, targets_high, max_shift),
        lambda: column_swap(aln, targets_high),
        lambda: move_gap_toward_target(aln, targets_low),   # explore only
    ]

    # Try operators in random order.
    random.shuffle(operators)
    for op in operators:
        cand = op()
        if cand != aln:
            return cand

    # If nothing worked and force, do a random gap shift.
    if force:
        all_cols = list(range(len(aln[0])))
        for _ in range(10):
            cand = block_gap_shift(aln, all_cols, max_shift)
            if cand != aln:
                return cand
            cand = move_gap_toward_target(aln, all_cols)
            if cand != aln:
                return cand

    return [s[:] for s in aln]