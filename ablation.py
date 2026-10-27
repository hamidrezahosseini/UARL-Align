"""
Ablation study for UARL-Align (reviewer comments 4 & 5).

═══════════════════════════════════════════════════════════════════════════
STRENGTHENED U-TARGETING VERSION
═══════════════════════════════════════════════════════════════════════════

Design of the RL agent: at each step, the agent selects
    • WHICH column to mutate
    • WHICH of 3 operators to apply (gap-move, column-swap, gap-shift)

Three improvements make uncertainty targeting a first-class contributor:

    1. Richer per-column features (10):
       entropy, gap fraction, informative fraction, left/right entropy,
       gap coherence, SP contribution, neighbor similarity,
       misalignment signal, conservation.

    2. Entropy-weighted, locally-attributed reward:
       r = 2·ΔSP·(1 + 3.5·U_col) + 4·ΔU_global + 3·ΔU_local + 2·is_new_best
       — a mutation in a high-U column receives up to 4.5× amplification;
       — a local-window U reduction (within ±3 columns) is directly rewarded.

    3. U-aware best tracking:
       An alignment is kept as "best" if (a) SP strictly improves, OR
       (b) SP is comparable (within −2) AND U is meaningfully lower (−0.005).

Variants (all share the MAFFT seed and the total mutation budget):

    V0  MAFFT seed only
    V1  Full UARL: RL(col+op) + U + EO
    V2  Fixed weights + U + EO (no RL)
    V3  Heuristic weights + U + EO (no RL)
    V4  Random weights + U + EO (no RL)
    V5  RL(col+op) with U features OFF + EO
    V6  RL(col+op) + U, no EO acceptance
    V7  MAFFT + EO (uniform columns)
    V8  MAFFT + entropy-guided, no EO
    V9  MAFFT + random walk (uniform columns, no EO)
"""

from __future__ import annotations

# ============================================================
# 1. VARIANTS — defined FIRST (no external dependencies)
# ============================================================

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    use_rl: bool
    use_uncertainty: bool
    use_eo: bool
    weight_mode: str  # "learned" | "fixed" | "heuristic" | "random"

    def __post_init__(self):
        if self.use_rl and self.weight_mode != "learned":
            raise ValueError(f"{self.name}: use_rl requires 'learned'")
        if (not self.use_rl) and self.weight_mode == "learned":
            raise ValueError(f"{self.name}: 'learned' requires use_rl")


VARIANTS: Dict[str, Variant] = {
    "V0_MAFFT_seed": Variant(
        "V0_MAFFT_seed", "Baseline: MAFFT seed only",
        False, False, False, "fixed"),
    "V1_Full_UARL": Variant(
        "V1_Full_UARL", "Full UARL: RL(col+op) + U + EO",
        True, True, True, "learned"),
    "V2_FixedWeights": Variant(
        "V2_FixedWeights", "Fixed weights + U + EO (no RL)",
        False, True, True, "fixed"),
    "V3_HeuristicWeights": Variant(
        "V3_HeuristicWeights", "Heuristic weights + U + EO (no RL)",
        False, True, True, "heuristic"),
    "V4_RandomWeights": Variant(
        "V4_RandomWeights", "Random weights + U + EO (no RL)",
        False, True, True, "random"),
    "V5_NoUncertainty": Variant(
        "V5_NoUncertainty", "RL(col+op) with U features OFF + EO",
        True, False, True, "learned"),
    "V6_NoEO": Variant(
        "V6_NoEO", "RL(col+op) + U, no EO acceptance",
        True, True, False, "learned"),
    "V7_EO_only": Variant(
        "V7_EO_only", "MAFFT + EO (fixed, uniform cols)",
        False, False, True, "fixed"),
    "V8_Uncertainty_only": Variant(
        "V8_Uncertainty_only", "MAFFT + U only (fixed, no EO)",
        False, True, False, "fixed"),
    "V9_Random_refine": Variant(
        "V9_Random_refine", "MAFFT + random walk (fixed, uniform)",
        False, False, False, "fixed"),
}


# ============================================================
# 2. Heavy imports
# ============================================================

import hashlib
import json
import math
import os
import random
import time
import types
from collections import Counter
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

import config as base_cfg
from fasta_io import read_fasta, write_fasta
from mafft_seed import run_mafft
from reward import compute_reward
from scoring import sum_of_pairs
from uncertainty import (
    alignment_uncertainty,
    column_uncertainties,
    infer_sequence_type,
)

# Legacy policy-gradient trainer (kept for backward compatibility,
# NOT used by the new RL implementation)
try:
    from pg_trainer import PGTrainer
    _PG_OK = True
    _PG_ERR = None
except Exception as _e:  # pragma: no cover
    _PG_OK = False
    _PG_ERR = _e
    PGTrainer = None


# ============================================================
# 3. Config helpers
# ============================================================

def make_cfg(**overrides) -> types.SimpleNamespace:
    """Create a per-run copy of the config."""
    cfg = types.SimpleNamespace()
    for name in dir(base_cfg):
        if name.isupper():
            setattr(cfg, name, getattr(base_cfg, name))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _stable_hash(s: str, mod: int = 10_000) -> int:
    """Deterministic hash independent of PYTHONHASHSEED."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


# ============================================================
# 4. Weight helpers (non-RL variants)
# ============================================================

def _heuristic_weights(col_unc, aln):
    """Plausible NON-LEARNING alternative to RL."""
    u_mean = float(np.mean(col_unc)) if col_unc else 0.0
    n, L = len(aln), len(aln[0]) if aln else 1
    g = sum(s.count("-") for s in aln)
    g_ratio = g / (n * L) if (n * L) > 0 else 0.0

    alpha = max(0.1, 1.0 - u_mean)
    beta = u_mean + 0.1
    gamma = min(1.0, 2.0 * g_ratio + 0.1)
    s = alpha + beta + gamma
    if s < 1e-8:
        return 1.0 / 3, 1.0 / 3, 1.0 / 3
    return alpha / s, beta / s, gamma / s


def _set_weights(cfg, mode, col_unc, aln, rng):
    """Populate cfg.ALPHA / BETA / GAMMA for non-RL variants."""
    if mode == "fixed":
        cfg.ALPHA, cfg.BETA, cfg.GAMMA = 1.0 / 3, 1.0 / 3, 1.0 / 3
    elif mode == "heuristic":
        cfg.ALPHA, cfg.BETA, cfg.GAMMA = _heuristic_weights(col_unc, aln)
    elif mode == "random":
        w = rng.dirichlet([1.0, 1.0, 1.0])
        cfg.ALPHA = float(w[0])
        cfg.BETA = float(w[1])
        cfg.GAMMA = float(w[2])


# ============================================================
# 5. Candidate validation
# ============================================================

def _validate_candidate(original, candidate):
    """Check that a mutation preserved sequence content and length."""
    if len(original) != len(candidate):
        raise ValueError("sequence count changed")
    n = len(original)
    if n == 0:
        return
    if len({len(s) for s in candidate}) != 1:
        raise ValueError("unequal row lengths")
    for i in range(n):
        if original[i].replace("-", "") != candidate[i].replace("-", ""):
            raise ValueError(f"residues changed in row {i}")


# ============================================================
# 6. Column-Operator Policy
# ============================================================

class ColumnOperatorPolicy(nn.Module):
    """
    Given per-column features (n_cols, 10), outputs:
        • col_logits : (n_cols,)
        • op_logits  : (n_cols, n_operators)

    The `sample` method supports an optional entropy prior (curriculum)
    that biases column selection toward high-uncertainty columns early
    in training.
    """

    def __init__(self, feature_dim: int = 10, hidden: int = 192,
                 n_operators: int = 3):
        super().__init__()
        self.n_operators = n_operators
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.col_head = nn.Linear(hidden, 1)
        self.op_head = nn.Linear(hidden, n_operators)

    def forward(self, feats: torch.Tensor):
        h = self.encoder(feats)
        col_logits = self.col_head(h).squeeze(-1)
        op_logits = self.op_head(h)
        return col_logits, op_logits

    def sample(self, feats: torch.Tensor, temperature: float = 1.0,
               entropy_prior=None, prior_alpha: float = 0.0):
        col_logits, op_logits = self.forward(feats)
        if entropy_prior is not None and prior_alpha > 0:
            prior = torch.tensor(
                entropy_prior, dtype=col_logits.dtype,
                device=col_logits.device,
            )
            col_logits = col_logits + prior_alpha * prior

        col_dist = torch.distributions.Categorical(
            logits=col_logits / temperature)
        col = col_dist.sample()
        op_dist = torch.distributions.Categorical(
            logits=op_logits[col] / temperature)
        op = op_dist.sample()
        log_prob = col_dist.log_prob(col) + op_dist.log_prob(op)
        return col.item(), op.item(), log_prob


# ============================================================
# 7. Column features  (10 per column)
# ============================================================

def compute_column_features(aln, use_uncertainty: bool = True):
    """
    Return (features, entropies).

        features : list[n_cols] of list[10]
        entropies: list[n_cols] with the raw normalized entropy

    Feature layout:
        [0] entropy
        [1] gap fraction
        [2] informative fraction
        [3] left-neighbor entropy
        [4] right-neighbor entropy
        [5] gap coherence  (1 = clustered, 0 = scattered)
        [6] SP contribution  (signed, normalized)
        [7] neighbor similarity (residue pattern match)
        [8] misalignment signal = entropy · (1−coherence) · (1−similarity)
        [9] conservation = 1 − entropy
    """
    n, L = len(aln), len(aln[0])

    # ---- 1. Per-column entropy ----
    entropies = []
    for j in range(L):
        col = [aln[i][j] for i in range(n)]
        non_gap = [c for c in col if c != "-"]
        if len(non_gap) >= 2:
            cnt = Counter(non_gap)
            total = len(non_gap)
            e = -sum((c / total) * math.log2(c / total)
                     for c in cnt.values())
            e = e / 2.0   # normalize by log2(4)
        else:
            e = 0.0
        entropies.append(e)

    # ---- 2. Gap coherence ----
    gap_coh = []
    for j in range(L):
        pattern = [1 if aln[i][j] == "-" else 0 for i in range(n)]
        n_gaps = sum(pattern)
        if n_gaps == 0:
            gap_coh.append(1.0)
        elif n_gaps == 1:
            gap_coh.append(0.5)
        else:
            runs = 0
            prev = 0
            for g in pattern:
                if g == 1 and prev == 0:
                    runs += 1
                prev = g
            coh = 1.0 - (runs - 1) / n_gaps
            gap_coh.append(max(0.0, coh))

    # ---- 3. SP contribution per column ----
    max_pair = (n * (n - 1) / 2.0) * 2.0
    sp_contrib = []
    for j in range(L):
        s = 0.0
        for i in range(n - 1):
            for k in range(i + 1, n):
                a, b = aln[i][j], aln[k][j]
                if a == "-" or b == "-":
                    s -= 1.0
                elif a == b:
                    s += 2.0
        sp_contrib.append(s / max(1.0, max_pair))

    # ---- 4. Neighbor similarity ----
    neighbor_sim = []
    for j in range(L):
        col = [aln[i][j] for i in range(n)]
        sims = []
        for dj in (-1, 1):
            k = j + dj
            if 0 <= k < L:
                ncol = [aln[i][k] for i in range(n)]
                match = sum(1 for a, b in zip(col, ncol)
                            if a == b and a != "-")
                sims.append(match / n)
        neighbor_sim.append(sum(sims) / len(sims) if sims else 0.0)

    # ---- 5. Assemble ----
    feats = []
    for j in range(L):
        col = [aln[i][j] for i in range(n)]
        n_gaps = col.count("-")
        non_gap = [c for c in col if c != "-"]

        entropy = entropies[j]
        gap_frac = n_gaps / n
        info_frac = len(non_gap) / n
        left = entropies[j - 1] if j > 0 else entropy
        right = entropies[j + 1] if j < L - 1 else entropy
        gc = gap_coh[j]
        sp = sp_contrib[j]
        ns = neighbor_sim[j]

        misalign_signal = entropy * (1.0 - gc) * (1.0 - ns)
        conservation = 1.0 - entropy

        if not use_uncertainty:
            entropy = 0.0
            left = 0.0
            right = 0.0
            gc = 0.0
            ns = 0.0
            misalign_signal = 0.0
            conservation = 0.0

        feats.append([
            entropy,           # 0
            gap_frac,          # 1
            info_frac,         # 2
            left,              # 3
            right,             # 4
            gc,                # 5
            sp,                # 6
            ns,                # 7
            misalign_signal,   # 8
            conservation,      # 9
        ])

    return feats, entropies


# ============================================================
# 8. RL mutation operators (3 types)
# ============================================================

def _move_gap_to_col(aln, col_idx, rng):
    """Move an existing gap to col_idx in a random eligible row."""
    n, L = len(aln), len(aln[0])
    eligible = []
    for i in range(n):
        if aln[i][col_idx] == "-":
            continue
        gaps = [k for k, c in enumerate(aln[i]) if c == "-"]
        if gaps:
            eligible.append((i, gaps))
    if not eligible:
        return aln
    i, gaps = eligible[int(rng.integers(0, len(eligible)))]
    g = int(rng.choice(gaps))
    if g == col_idx:
        return aln
    row = list(aln[i])
    row.pop(g)
    row.insert(col_idx, "-")
    if "".join(row).replace("-", "") != aln[i].replace("-", ""):
        return aln
    out = list(aln)
    out[i] = "".join(row)
    return out


def _swap_column(aln, col_idx, rng):
    """Swap col_idx with a neighbor (±1)."""
    L = len(aln[0])
    if L < 2:
        return aln
    delta = int(rng.choice([-1, 1]))
    other = col_idx + delta
    if not (0 <= other < L):
        return aln
    a, b = min(col_idx, other), max(col_idx, other)
    return [s[:a] + s[b] + s[a + 1:b] + s[a] + s[b + 1:]
            for s in aln]


def _shift_gap_near(aln, col_idx, rng, window=3):
    """Shift a gap within ±window of col_idx by ±1-2."""
    n, L = len(aln), len(aln[0])
    candidates = []
    for i in range(n):
        for j in range(max(0, col_idx - window),
                       min(L, col_idx + window + 1)):
            if aln[i][j] == "-":
                candidates.append((i, j))
    if not candidates:
        return aln
    i, j = candidates[int(rng.integers(0, len(candidates)))]
    delta = int(rng.integers(-2, 3))
    if delta == 0:
        delta = 1
    new_j = max(0, min(L - 1, j + delta))
    if new_j == j:
        return aln
    row = list(aln[i])
    row.pop(j)
    row.insert(new_j, "-")
    if "".join(row).replace("-", "") != aln[i].replace("-", ""):
        return aln
    out = list(aln)
    out[i] = "".join(row)
    return out


def apply_rl_mutation(aln, col_idx, op, rng):
    """Dispatch to one of three operators."""
    if op == 0:
        return _move_gap_to_col(aln, col_idx, rng)
    if op == 1:
        return _swap_column(aln, col_idx, rng)
    if op == 2:
        return _shift_gap_near(aln, col_idx, rng)
    return aln


# ============================================================
# 9. Local entropy change (windowed)
# ============================================================

def _local_entropy_change(aln_old, aln_new, col, window=3):
    """
    Mean entropy change in a ±window neighborhood around `col`.
    Positive = uncertainty decreased locally.
    """
    n, L = len(aln_old), len(aln_old[0])
    lo = max(0, col - window)
    hi = min(L, col + window + 1)
    if hi <= lo:
        return 0.0

    def _window_entropy(aln):
        total, count = 0.0, 0
        for j in range(lo, hi):
            chars = [aln[i][j] for i in range(n)]
            non_gap = [c for c in chars if c != "-"]
            if len(non_gap) >= 2:
                cnt = Counter(non_gap)
                T = len(non_gap)
                e = -sum((k / T) * math.log2(k / T)
                         for k in cnt.values()) / 2.0
                total += e
                count += 1
        return total / count if count > 0 else 0.0

    return _window_entropy(aln_old) - _window_entropy(aln_new)


# ============================================================
# 10. RL training loop (STRENGTHENED U-targeting)
# ============================================================

def run_variant_rl_new(variant, seed_alignment, seqs, run_seed,
                       cfg, budget, verbose=False):
    """
    RL with strong uncertainty targeting.

    Reward (per step):
        r = 2 · ΔSP · (1 + 3.5·U_col)     # entropy-weighted SP gain
          + 4 · ΔU_global                  # global entropy reduction
          + 3 · ΔU_local                   # local (windowed) entropy reduction
          + 2 · 1[new best]                # bonus for new SP record
        clipped to [−10, +10].

    Best tracking (U-aware):
        keep as "best" if SP strictly improves, OR
        if SP is comparable (−2) AND U is meaningfully lower (−0.005).

    Curriculum prior on the policy:
        col_logits += α · entropy   with α: 2.5 → 0
    """
    seed_alignment = [s.upper() for s in seed_alignment]
    seqs = [s.upper() for s in seqs]

    if budget <= 0:
        return list(seed_alignment), {
            "n_mutations": 0, "n_policy_updates": 0}

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)

    try:
        seq_type = infer_sequence_type(seqs)
    except Exception:
        seq_type = "auto"

    use_uncertainty = variant.use_uncertainty
    use_eo = variant.use_eo

    # ---- Policy ----
    policy = ColumnOperatorPolicy(feature_dim=10, hidden=192,
                                   n_operators=3)
    lr = float(getattr(cfg, "RL_LR", 0.005))
    lr = max(1e-4, min(1e-2, lr))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = 0.0

    # ---- State ----
    current = list(seed_alignment)
    best = list(seed_alignment)
    best_sp = sum_of_pairs(seed_alignment)
    try:
        best_u = alignment_uncertainty(seed_alignment,
                                        sequence_type=seq_type)
    except Exception:
        best_u = 1.0

    n_mutations = 0
    n_policy_updates = 0

    episode_len = 50
    n_episodes = max(1, budget // episode_len)

    stall_counter = 0
    stall_threshold = max(300, budget // 25)

    eps_start, eps_end = 0.7, 0.05

    for ep in range(n_episodes):
        progress = ep / max(1, n_episodes - 1)

        epsilon = eps_start * (1.0 - progress) + eps_end * progress
        temperature = max(0.3, 1.5 * (1.0 - progress))
        # Stronger prior (2.5 → 0) vs. previous (1.5 → 0)
        prior_alpha = (2.5 * (1.0 - progress)) if use_uncertainty else 0.0

        ep_log_probs = []
        ep_rewards = []

        for step in range(episode_len):
            if n_mutations >= budget:
                break

            feats, entropies = compute_column_features(
                current, use_uncertainty)
            feats_t = torch.tensor(feats, dtype=torch.float32)

            use_heuristic = (rng.random() < epsilon)

            if use_heuristic:
                if use_uncertainty:
                    w = np.array(entropies, dtype=float) + 1e-8
                    w = w / w.sum()
                    col = int(rng.choice(len(entropies), p=w))
                else:
                    col = int(rng.integers(0, len(entropies)))
                op = int(rng.integers(0, 3))
                log_prob = None
            else:
                col, op, log_prob = policy.sample(
                    feats_t, temperature,
                    entropy_prior=(entropies if use_uncertainty else None),
                    prior_alpha=prior_alpha,
                )

            col_entropy_before = entropies[col]

            cand = apply_rl_mutation(current, col, op, rng)
            n_mutations += 1

            if cand == current:
                if log_prob is not None:
                    ep_log_probs.append(log_prob)
                    ep_rewards.append(-0.5)
                stall_counter += 1
                continue

            try:
                _validate_candidate(current, cand)
            except ValueError:
                if log_prob is not None:
                    ep_log_probs.append(log_prob)
                    ep_rewards.append(-0.5)
                stall_counter += 1
                continue

            # ---- ΔSP, ΔU (global), ΔU (local) ----
            sp_old = sum_of_pairs(current)
            sp_new = sum_of_pairs(cand)
            delta_sp = sp_new - sp_old

            try:
                u_old = alignment_uncertainty(
                    current, sequence_type=seq_type)
                u_new = alignment_uncertainty(
                    cand, sequence_type=seq_type)
                delta_u = u_old - u_new
            except Exception:
                delta_u = 0.0

            if use_uncertainty:
                local_delta_u = _local_entropy_change(
                    current, cand, col, window=3)
            else:
                local_delta_u = 0.0

            is_new_best = 1.0 if sp_new > best_sp else 0.0

            # ---- Entropy-weighted reward ----
            if use_uncertainty:
                u_weight = 1.0 + 3.5 * col_entropy_before
                weighted_delta_sp = delta_sp * u_weight
            else:
                weighted_delta_sp = delta_sp

            reward = (
                2.0 * weighted_delta_sp
                + 4.0 * delta_u
                + 3.0 * local_delta_u
                + 2.0 * is_new_best
            )
            reward = max(-10.0, min(10.0, reward))

            if log_prob is not None:
                ep_log_probs.append(log_prob)
                ep_rewards.append(reward)

            # ---- Acceptance ----
            if use_eo:
                accept = sp_new >= sp_old
            else:
                accept = True

            if accept:
                current = cand

                # ---- U-aware best tracking ----
                if sp_new > best_sp + 1.0:
                    best = list(current)
                    best_sp = sp_new
                    best_u = u_new
                    stall_counter = 0
                elif (sp_new > best_sp - 2.0
                      and u_new < best_u - 0.005):
                    best = list(current)
                    best_sp = sp_new
                    best_u = u_new
                    stall_counter = 0
                else:
                    stall_counter += 1

            if stall_counter >= stall_threshold:
                current = list(best)
                stall_counter = 0

        # ---- Policy update ----
        if ep_log_probs:
            returns = np.zeros(len(ep_rewards))
            R = 0.0
            for k in range(len(ep_rewards) - 1, -1, -1):
                R = ep_rewards[k] + 0.95 * R
                returns[k] = R

            adv = torch.tensor(returns - baseline, dtype=torch.float32)
            loss = sum(-lp * a for lp, a in zip(ep_log_probs, adv))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            baseline = 0.9 * baseline + 0.1 * float(np.mean(returns))
            n_policy_updates += 1

        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"    ep {ep+1}/{n_episodes}  best_SP={best_sp:.1f}  "
                  f"best_U={best_u:.4f}  eps={epsilon:.2f}  "
                  f"prior={prior_alpha:.2f}")

    return best, {
        "n_mutations": n_mutations,
        "n_policy_updates": n_policy_updates,
    }


# ============================================================
# 11. Best-of-N wrapper
# ============================================================

def run_variant_rl_best_of_n(variant, seed_alignment, seqs,
                              run_seed, cfg, budget,
                              n_seeds=5, verbose=False):
    """
    Run RL N times with different seeds; keep the best alignment.
    Total budget is preserved: each seed gets budget // n_seeds.
    """
    best = None
    best_sp = -float("inf")
    best_info = None

    per_seed_budget = max(100, budget // n_seeds)

    for i in range(n_seeds):
        seed_i = run_seed + i * 9973
        result, info = run_variant_rl_new(
            variant, seed_alignment, seqs, seed_i,
            cfg, per_seed_budget, verbose=False)
        sp = sum_of_pairs(result)
        if verbose:
            print(f"    seed {i+1}/{n_seeds}: SP={sp:.1f}")
        if sp > best_sp:
            best_sp = sp
            best = result
            best_info = info

    best_info = dict(best_info or {})
    best_info["n_mutations"] = per_seed_budget * n_seeds
    best_info["n_seeds"] = n_seeds
    return best, best_info


# ============================================================
# 12. Dispatcher
# ============================================================

def run_variant(variant, seed_alignment, seqs, run_seed,
                cfg, budget: int = 10000) -> Tuple[List[str], Dict]:
    """
    Dispatch to the correct implementation.
    RL variants optionally use best-of-N (cfg.USE_BEST_OF_N).
    """
    if variant.use_rl:
        use_bon = bool(getattr(cfg, "USE_BEST_OF_N", True))
        n = int(getattr(cfg, "BEST_OF_N", 5))
        if use_bon and n > 1:
            return run_variant_rl_best_of_n(
                variant, seed_alignment, seqs, run_seed,
                cfg, budget, n_seeds=n, verbose=False)
        return run_variant_rl_new(
            variant, seed_alignment, seqs, run_seed, cfg, budget)

    # ---- Non-RL path ----
    from mutations import adaptive_mutation

    seed_alignment = [s.upper() for s in seed_alignment]
    seqs = [s.upper() for s in seqs]

    if variant.name == "V0_MAFFT_seed":
        return list(seed_alignment), {
            "n_mutations": 0, "n_policy_updates": 0}

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)

    try:
        seq_type = infer_sequence_type(seqs)
    except Exception:
        seq_type = "auto"

    current = list(seed_alignment)
    best = list(seed_alignment)
    best_sp = sum_of_pairs(seed_alignment)
    n_mutations = 0
    n_search = max(1, budget)

    for step in range(n_search):
        try:
            col_unc = column_uncertainties(
                current, sequence_type=seq_type)
        except Exception:
            col_unc = [0.5] * len(current[0])

        if not variant.use_uncertainty:
            col_unc = [1.0] * len(col_unc)

        phase = "explore" if step < n_search // 2 else "exploit"
        _set_weights(cfg, variant.weight_mode, col_unc, current, rng)

        try:
            cand = adaptive_mutation(
                current, col_unc, cfg, phase=phase,
                force=(step % 10 == 0))
        except Exception:
            continue
        n_mutations += 1

        if cand == current:
            continue

        try:
            _validate_candidate(current, cand)
        except ValueError:
            continue

        sp_old = sum_of_pairs(current)
        sp_new = sum_of_pairs(cand)

        if variant.use_eo:
            try:
                reward, _ = compute_reward(current, cand, cfg)
            except Exception:
                reward = 0.0
            accept = ((sp_new > sp_old) or (reward > 0)
                      or (random.random() < 0.05))
        else:
            accept = True

        if accept:
            current = cand
            if sp_new > best_sp:
                best_sp = sp_new
                best = list(current)

    return best, {"n_mutations": n_mutations, "n_policy_updates": 0}


# ============================================================
# 13. Metrics
# ============================================================

def _em_cols(aln):
    """Count fully conserved, gap-free columns."""
    n, L = len(aln), len(aln[0])
    em = 0
    for j in range(L):
        c0 = aln[0][j]
        if c0 == "-":
            continue
        if all(aln[i][j] == c0 for i in range(n)):
            em += 1
    return em


def _internal_cs(aln):
    if not aln or not aln[0]:
        return 0.0
    return _em_cols(aln) / len(aln[0])


def _pos_maps(aln):
    maps = []
    for s in aln:
        m, idx = {}, 0
        for c, ch in enumerate(s):
            if ch != "-":
                m[idx] = c
                idx += 1
        maps.append(m)
    return maps


def _sps(pred, ref):
    n = len(pred)
    if n != len(ref):
        return float("nan")
    pm_p = _pos_maps(pred)
    pm_r = _pos_maps(ref)
    num, den = 0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            rp = {(a, b) for a, ca in pm_r[i].items()
                  for b, cb in pm_r[j].items() if ca == cb}
            pp = {(a, b) for a, ca in pm_p[i].items()
                  for b, cb in pm_p[j].items() if ca == cb}
            num += len(rp & pp)
            den += len(rp)
    return num / den if den > 0 else 0.0


def _tc(pred, ref):
    n = len(pred)
    if n != len(ref):
        return float("nan")
    Lp, Lr = len(pred[0]), len(ref[0])
    pm_r = _pos_maps(ref)
    correct = 0
    for c in range(Lp):
        rc_set, ok = set(), True
        for i in range(n):
            if pred[i][c] == "-":
                continue
            oi = sum(1 for x in pred[i][:c] if x != "-")
            if oi not in pm_r[i]:
                ok = False
                break
            rc_set.add(pm_r[i][oi])
        if not ok or len(rc_set) != 1:
            continue
        rc = next(iter(rc_set))
        if all((pred[i][c] == "-") == (ref[i][rc] == "-")
               for i in range(n)):
            correct += 1
    return correct / Lr if Lr > 0 else 0.0


def compute_metrics(aln, seq_type, ref=None):
    try:
        u = alignment_uncertainty(aln, sequence_type=seq_type)
    except Exception:
        u = float("nan")
    row = {
        "SP": sum_of_pairs(aln),
        "CS": _internal_cs(aln),
        "U": u,
        "AL": len(aln[0]) if aln else 0,
    }
    if ref is not None:
        row["SPS"] = _sps(aln, ref)
        row["TC"] = _tc(aln, ref)
    return row


# ============================================================
# 14. File I/O
# ============================================================

def find_fasta_files(input_dir):
    files = []
    for pat in ("*.fasta", "*.fa", "*.fna", "*.fas"):
        files.extend(glob(os.path.join(input_dir, pat)))
    return sorted(set(files))


def find_reference(reference_dir, dataset_name):
    if not reference_dir:
        return None
    for ext in (".aln", ".clustal", ".fasta", ".fa"):
        p = os.path.join(reference_dir, dataset_name + ext)
        if os.path.exists(p):
            return p
    return None


def read_reference_alignment(path):
    try:
        from Bio import AlignIO
        ext = os.path.splitext(path)[1].lower()
        fmt = "clustal" if ext in (".aln", ".clustal") else "fasta"
        return [str(r.seq).upper() for r in AlignIO.read(path, fmt)]
    except Exception:
        seqs, cur = [], ""
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if cur:
                        seqs.append(cur)
                        cur = ""
                elif line and not line.startswith(" "):
                    parts = line.split()
                    if parts:
                        cur += parts[-1]
        if cur:
            seqs.append(cur)
        return [s.upper() for s in seqs]


# ============================================================
# 15. Batch processing
# ============================================================

def process_dataset(fasta_path, variants, cfg, output_dir,
                    reference_path=None, budget=10000, verbose=True):
    dataset_name = Path(fasta_path).stem
    output_dir = Path(output_dir)
    aln_dir = output_dir / "alignments" / dataset_name
    aln_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 70}\n[{dataset_name}] {fasta_path}")

    seqs = read_fasta(fasta_path)
    if len(seqs) < 2:
        print(f"[{dataset_name}] SKIP: <2 sequences")
        return []

    n_seqs = len(seqs)
    try:
        seq_type = infer_sequence_type(seqs)
    except Exception:
        seq_type = "auto"

    if verbose:
        print(f"[{dataset_name}] n={n_seqs}  type={seq_type}  "
              f"len∈[{min(len(s) for s in seqs)},"
              f"{max(len(s) for s in seqs)}]")

    reference = None
    if reference_path:
        try:
            reference = read_reference_alignment(reference_path)
            if len(reference) != n_seqs:
                print(f"[{dataset_name}] Ref count mismatch; skip SPS/TC")
                reference = None
        except Exception as e:
            print(f"[{dataset_name}] Ref failed: {e}")

    t0 = time.time()
    try:
        seed = run_mafft(seqs, cfg.MAFFT_CMD)
        t_mafft = time.time() - t0
        if verbose:
            print(f"[{dataset_name}] MAFFT: {t_mafft:.2f}s")
    except Exception as e:
        print(f"[{dataset_name}] MAFFT failed ({e}); abort.")
        return []

    if not seed or len({len(s) for s in seed}) != 1:
        print(f"[{dataset_name}] invalid seed; abort.")
        return []

    write_fasta(seed,
                str(aln_dir / f"{dataset_name}_V0_MAFFT_seed.fasta"))

    base_m = compute_metrics(seed, seq_type, reference)
    base_m.update({
        "dataset": dataset_name,
        "variant": "V0_MAFFT_seed",
        "description": VARIANTS["V0_MAFFT_seed"].description,
        "n_seqs": n_seqs,
        "n_mutations": 0,
        "n_policy_updates": 0,
        "runtime_s": round(t_mafft, 3),
        "delta_SP_vs_seed": 0.0,
        "delta_SPS_vs_seed": 0.0 if reference else float("nan"),
        "delta_TC_vs_seed": 0.0 if reference else float("nan"),
    })
    results = [base_m]

    sp_seed = base_m["SP"]
    sps_seed = base_m.get("SPS", float("nan"))
    tc_seed = base_m.get("TC", float("nan"))

    for vname, variant in variants.items():
        if vname == "V0_MAFFT_seed":
            continue

        if verbose:
            print(f"[{dataset_name}] {vname} ... ", end="", flush=True)

        base_seed = (cfg.RANDOM_SEED
                     if getattr(cfg, "RANDOM_SEED", None) is not None
                     else 42)
        run_seed = (base_seed
                    + _stable_hash(vname, 10_000)
                    + _stable_hash(dataset_name, 1_000))

        try:
            t0 = time.time()
            refined, info = run_variant(
                variant, seed, seqs, run_seed, cfg, budget=budget)
            runtime = time.time() - t0
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")
            import traceback
            traceback.print_exc()
            continue

        if not refined or len({len(s) for s in refined}) != 1:
            print("invalid output; skip")
            continue

        write_fasta(refined,
                    str(aln_dir / f"{dataset_name}_{vname}.fasta"))

        m = compute_metrics(refined, seq_type, reference)
        row = {
            "dataset": dataset_name,
            "variant": vname,
            "description": variant.description,
            "n_seqs": n_seqs,
            "n_mutations": info["n_mutations"],
            "n_policy_updates": info["n_policy_updates"],
            "runtime_s": round(runtime, 3),
            "delta_SP_vs_seed": m["SP"] - sp_seed,
            "delta_SPS_vs_seed": (
                m.get("SPS", float("nan")) - sps_seed
                if "SPS" in m else float("nan")),
            "delta_TC_vs_seed": (
                m.get("TC", float("nan")) - tc_seed
                if "TC" in m else float("nan")),
        }
        row.update(m)
        results.append(row)

        if verbose:
            print(f"SP={m['SP']:.2f} U={m['U']:.3f} "
                  f"mut={info['n_mutations']} t={runtime:.2f}s")

    return results


def process_all(input_dir, output_dir, cfg, variants,
                reference_dir=None, budget=10000, verbose=True):
    files = find_fasta_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No FASTA files in {input_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignments").mkdir(exist_ok=True)

    if verbose:
        print(f"Datasets: {len(files)}  |  Variants: {len(variants)}")
        print(f"Budget: {budget} mutation calls / variant")

    all_rows = []
    t_start = time.time()

    for f in files:
        dname = Path(f).stem
        ref = find_reference(reference_dir, dname)
        try:
            rows = process_dataset(
                f, variants, cfg, str(output_dir),
                reference_path=ref, budget=budget, verbose=verbose)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[{dname}] FATAL: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t_start

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "results.csv", index=False)

    if not df.empty:
        metric_cols = [
            c for c in (
                "SP", "CS", "U", "SPS", "TC",
                "delta_SP_vs_seed", "delta_SPS_vs_seed",
                "delta_TC_vs_seed", "runtime_s", "n_mutations",
            ) if c in df.columns
        ]
        summary = df.groupby("variant")[metric_cols].agg(["mean", "std"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary = summary.round(4)
        summary = summary.reindex(
            [v for v in VARIANTS if v in summary.index])
        summary.to_csv(output_dir / "summary.csv")

        if verbose:
            print(f"\n{'=' * 70}\n=== Mean per variant ===")
            show = [c for c in ("SP_mean", "CS_mean", "U_mean",
                                "SPS_mean", "TC_mean")
                    if c in summary.columns]
            print(summary[show].to_string())

    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_datasets": len(files),
        "budget_per_variant": budget,
        "variants": list(variants.keys()),
        "total_runtime_s": round(elapsed, 2),
        "config": {k: v for k, v in vars(cfg).items()
                   if k.isupper() and not k.endswith("_CMD")},
    }
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    return df


# ============================================================
# 16. Statistical analysis
# ============================================================

def paired_tests(results_csv, baseline="V1_Full_UARL"):
    df = pd.read_csv(results_csv)
    metrics = [m for m in ("SP", "CS", "U", "SPS", "TC")
               if m in df.columns]
    rows = []
    for var in [v for v in df["variant"].unique() if v != baseline]:
        for metric in metrics:
            pivot = df.pivot_table(
                index="dataset", columns="variant", values=metric)
            if baseline not in pivot.columns or var not in pivot.columns:
                continue
            common = pivot[[baseline, var]].dropna()
            if len(common) < 3:
                continue
            a, b = common[baseline].values, common[var].values
            try:
                _stat, p = wilcoxon(a, b)
            except Exception:
                continue
            rows.append({
                "variant": var, "metric": metric,
                "n_datasets": len(common),
                "mean_baseline": round(float(a.mean()), 4),
                "mean_variant": round(float(b.mean()), 4),
                "delta": round(float(a.mean() - b.mean()), 4),
                "p_value": round(float(p), 6),
                "significant": bool(p < 0.05),
                "favors": ("baseline"
                           if (metric == "U" and a.mean() < b.mean())
                           or (metric != "U" and a.mean() > b.mean())
                           else "variant"),
            })
    return pd.DataFrame(rows)


def component_contributions(results_csv):
    df = pd.read_csv(results_csv)
    pairs = {
        "RL_vs_fixed":   ("V1_Full_UARL", "V2_FixedWeights"),
        "RL_vs_heur":    ("V1_Full_UARL", "V3_HeuristicWeights"),
        "RL_vs_rand":    ("V1_Full_UARL", "V4_RandomWeights"),
        "Uncertainty":   ("V1_Full_UARL", "V5_NoUncertainty"),
        "EO":            ("V1_Full_UARL", "V6_NoEO"),
        "Overall_gain":  ("V1_Full_UARL", "V0_MAFFT_seed"),
        "vs_EO_only":    ("V1_Full_UARL", "V7_EO_only"),
        "vs_U_only":     ("V1_Full_UARL", "V8_Uncertainty_only"),
        "vs_random_ref": ("V1_Full_UARL", "V9_Random_refine"),
    }
    metrics = [m for m in ("SP", "CS", "U", "SPS", "TC")
               if m in df.columns]
    rows = []
    for label, (full, base) in pairs.items():
        for metric in metrics:
            pivot = df.pivot_table(
                index="dataset", columns="variant", values=metric)
            if full not in pivot.columns or base not in pivot.columns:
                continue
            common = pivot[[full, base]].dropna()
            if common.empty:
                continue
            fm, bm = common[full].mean(), common[base].mean()
            delta = fm - bm
            improvement = -delta if metric == "U" else delta
            rows.append({
                "comparison": label, "full": full, "baseline": base,
                "metric": metric,
                "full_mean": round(float(fm), 4),
                "base_mean": round(float(bm), 4),
                "improvement": round(float(improvement), 4),
            })
    return pd.DataFrame(rows)


def win_rates(results_csv):
    df = pd.read_csv(results_csv)
    metrics = [m for m in ("SP", "CS", "U", "SPS", "TC")
               if m in df.columns]
    rows = []
    for metric in metrics:
        pivot = df.pivot_table(
            index="dataset", columns="variant", values=metric)
        if pivot.empty:
            continue
        best_idx = (pivot.idxmin(axis=1) if metric == "U"
                    else pivot.idxmax(axis=1))
        counts = best_idx.value_counts()
        for var in pivot.columns:
            rows.append({
                "variant": var, "metric": metric,
                "wins": int(counts.get(var, 0)),
                "n_datasets": len(pivot),
                "win_rate": round(counts.get(var, 0) / len(pivot), 3),
            })
    return pd.DataFrame(rows)


def build_report(results_csv, output_dir=None):
    results_csv = Path(results_csv)
    df = pd.read_csv(results_csv)
    tests = paired_tests(results_csv)
    contrib = component_contributions(results_csv)
    wins = win_rates(results_csv)

    if output_dir:
        output_dir = Path(output_dir)
        tests.to_csv(output_dir / "statistical_tests.csv", index=False)
        contrib.to_csv(output_dir / "component_contributions.csv",
                       index=False)
        wins.to_csv(output_dir / "win_rates.csv", index=False)

    lines = []
    lines.append("=" * 75)
    lines.append("ABLATION STUDY REPORT  (reviewer comments 4 & 5)")
    lines.append("=" * 75)
    lines.append(f"\nDatasets: {df['dataset'].nunique()}")
    lines.append(f"Variants: {list(VARIANTS.keys())}")

    lines.append("\n" + "-" * 75)
    lines.append("1. COMPONENT CONTRIBUTIONS (Δ of V1 vs each ablation)")
    lines.append("-" * 75)
    if not contrib.empty:
        for label in contrib["comparison"].unique():
            sub = contrib[contrib["comparison"] == label]
            lines.append(f"\n  {label}:")
            for _, r in sub.iterrows():
                arrow = "↑" if r["improvement"] > 0 else "↓"
                lines.append(
                    f"    {r['metric']:<4}: "
                    f"V1={r['full_mean']:.4f}  vs  "
                    f"{r['baseline']}={r['base_mean']:.4f}  "
                    f"→ {arrow}{abs(r['improvement']):.4f}")

    lines.append("\n" + "-" * 75)
    lines.append("2. PAIRED WILCOXON TESTS vs V1_Full_UARL")
    lines.append("-" * 75)
    if not tests.empty:
        sig = tests[tests["significant"]]
        if sig.empty:
            lines.append("  No significant differences detected.")
        else:
            for _, r in sig.iterrows():
                lines.append(
                    f"  {r['variant']:<22} {r['metric']:<4}  "
                    f"p={r['p_value']:.4f}  favors={r['favors']}")

    lines.append("\n" + "-" * 75)
    lines.append("3. WIN RATES")
    lines.append("-" * 75)
    if not wins.empty:
        for metric in wins["metric"].unique():
            top = wins[wins["metric"] == metric].sort_values(
                "win_rate", ascending=False).head(3)
            lines.append(f"  {metric:<4}: " + ", ".join(
                f"{r['variant']}={r['win_rate']:.2f}"
                for _, r in top.iterrows()))

    lines.append("\n" + "-" * 75)
    lines.append("4. TAKE-AWAYS FOR REVIEWERS")
    lines.append("-" * 75)

    def _d(label, metric):
        sub = contrib[(contrib["comparison"] == label)
                       & (contrib["metric"] == metric)]
        return float(sub["improvement"].iloc[0]) if not sub.empty else None

    checks = [
        ("RL_vs_fixed", "SP", "RL vs fixed weights"),
        ("RL_vs_heur", "SP", "RL vs heuristic weights"),
        ("RL_vs_rand", "SP", "RL vs random weights"),
        ("Uncertainty", "SP", "Uncertainty targeting (SP)"),
        ("Uncertainty", "U", "Uncertainty targeting (U)"),
        ("EO", "SP", "EO search"),
        ("Overall_gain", "SP", "Overall gain vs MAFFT seed"),
        ("vs_random_ref", "SP", "vs random refinement (same budget)"),
    ]
    for label, metric, desc in checks:
        d = _d(label, metric)
        if d is None:
            continue
        thr = 0 if metric == "U" else 0.001
        verdict = "POSITIVE" if d > thr else "marginal/negative"
        lines.append(f"  • {desc:<40} {d:+.4f}  ({verdict})")

    lines.append("\n" + "=" * 75)
    report = "\n".join(lines)
    if output_dir:
        (Path(output_dir) / "ablation_report.txt").write_text(
            report, encoding="utf-8")
    return report