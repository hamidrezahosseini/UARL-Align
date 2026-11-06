"""
analyze_reward_weights.py
Section 5.3.10 — Analysis of Effective Reward Signals

Runs the UARL-Align policy on the 7 ablation datasets across 5 seeds
(42, 43, 44, 45, 46), records a per-episode summary of the policy's
behavior, and produces:

    output_weights/
    ├── Table28.csv              — phase-wise means, normalized shares, CV
    ├── Table28.md               — markdown for direct paste into Word
    ├── Figure9.png              — 3-panel trajectory plot with T_switch
    ├── per_episode_weights.csv  — full raw log
    └── phasewise_means.csv      — per (dataset, seed, phase) means

JUST PRESS F5 IN IDLE. No arguments needed.

Design note
-----------
The current UARL-Align (ablation.py) uses a (column, operator) action
space. The reward weights in Eq. (31) are fixed constants. To
characterize the *effective trade-off* the policy learns, three
behavioral signals are recorded per episode:

    alpha_eff = mean over steps of |ΔSP| / max(1, |SP_old|)
    beta_eff  = mean over steps of the entropy U_j of the selected column
    gamma_eff = mean over steps of (fraction of positions whose gap
                                    status changed)

These are the closest behavioral analogues of (alpha, beta, gamma) in
the current architecture. If a weight-learning policy is present
(pg_trainer.PGTrainer with `sample_weights`), the script auto-detects
it and uses the true (alpha, beta, gamma) instead.
"""

from __future__ import annotations

import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports (must run from project root)
# ---------------------------------------------------------------------------
try:
    import config as base_cfg
    from fasta_io import read_fasta
    from mafft_seed import run_mafft
    from scoring import sum_of_pairs
    from uncertainty import alignment_uncertainty, infer_sequence_type
    from ablation import (
        make_cfg,
        ColumnOperatorPolicy,
        compute_column_features,
        apply_rl_mutation,
    )
except ImportError as e:
    print(f"\n[FATAL] Import failed: {e}")
    print("Run this script from the project root "
          "(the folder that contains ablation.py, config.py, etc.).")
    sys.exit(1)

# Optional: legacy weight-learning policy
_WEIGHT_POLICY_AVAILABLE = False
PGTrainer = None
try:
    from pg_trainer import PGTrainer as _PG
    if hasattr(_PG, "sample_weights"):
        PGTrainer = _PG
        _WEIGHT_POLICY_AVAILABLE = True
except Exception:
    pass


# ===========================================================================
# USER-EDITABLE DEFAULTS  (change these to customize without CLI args)
# ===========================================================================

DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output_weights"

DEFAULT_DATASETS = [
    "1aboA_ref1",
    "1fjlA_ref1",
    "1hfh_ref1",
    "1idy_ref1",
    "1pfc_ref1",
    "1tgxA_ref1",
    "RNAstrAlign_group_I_intron_database__IA1",
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]

DEFAULT_BUDGET = 10000       # mutations per (dataset, seed)
EPISODE_LEN = 50              # steps per episode

QUICK_MODE = False            # set True for a fast smoke test (3 seeds × 500)

# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fast_gap_changed_fraction(aln_old, aln_new):
    """Fraction of positions whose gap status changed (fast, pure Python)."""
    n = len(aln_old)
    L = len(aln_old[0])
    total = 0
    changed = 0
    for i in range(n):
        a = aln_old[i]
        b = aln_new[i]
        for j in range(L):
            total += 1
            if (a[j] == "-") != (b[j] == "-"):
                changed += 1
    return changed / total if total else 0.0


def _mean_or_zero(lst):
    return float(np.mean(lst)) if lst else 0.0


# ---------------------------------------------------------------------------
# Core: run one (dataset, seed), record per-episode behavior
# ---------------------------------------------------------------------------
def run_one(fasta_path, seed, cfg, budget, verbose=True):
    """Train UARL-Align for one (dataset, seed); return per-episode DataFrame."""
    seqs = read_fasta(fasta_path)
    seed_aln = run_mafft(seqs, cfg.MAFFT_CMD)
    try:
        seq_type = infer_sequence_type(seqs)
    except Exception:
        seq_type = "auto"

    # ---- reproducibility ----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ---- policy ----
    if _WEIGHT_POLICY_AVAILABLE:
        policy = PGTrainer(cfg)
        optimizer = None
        policy_type = "weights"
    else:
        policy = ColumnOperatorPolicy(feature_dim=10, hidden=192,
                                       n_operators=3)
        lr = float(getattr(cfg, "RL_LR", 0.005))
        lr = max(1e-4, min(1e-2, lr))
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        policy_type = "column_op"

    # ---- state ----
    current = list(seed_aln)
    best = list(seed_aln)
    best_sp = sum_of_pairs(seed_aln)

    n_episodes = max(1, budget // EPISODE_LEN)
    n_steps = 0
    stall = 0
    stall_threshold = max(300, budget // 25)

    rows = []

    for ep in range(n_episodes):
        progress = ep / max(1, n_episodes - 1)
        epsilon = 0.7 * (1.0 - progress) + 0.05 * progress
        temperature = max(0.3, 1.5 * (1.0 - progress))
        prior_alpha = 2.5 * (1.0 - progress)

        ep_log_probs = []
        ep_rewards = []
        step_alpha = []
        step_beta = []
        step_gamma = []
        step_true_weights = []

        for _ in range(EPISODE_LEN):
            if n_steps >= budget:
                break

            if policy_type == "weights":
                # ----- Legacy weight-learning mode -----
                a, b, c, log_prob = policy.sample_weights(current)
                step_true_weights.append((a, b, c))
                try:
                    from mutations import adaptive_mutation
                    col_unc = [0.5] * len(current[0])
                    cand = adaptive_mutation(
                        current, col_unc, cfg,
                        phase=("explore" if ep < n_episodes // 2 else "exploit"),
                        force=False,
                    )
                except Exception:
                    cand = current
                entropy_before = 0.0
            else:
                # ----- Column-operator mode (default) -----
                feats, entropies = compute_column_features(current, True)
                feats_t = torch.tensor(feats, dtype=torch.float32)

                use_heuristic = rng.random() < epsilon
                if use_heuristic:
                    w = np.array(entropies, dtype=float) + 1e-8
                    w = w / w.sum()
                    col = int(rng.choice(len(entropies), p=w))
                    op = int(rng.integers(0, 3))
                    log_prob = None
                else:
                    col, op, log_prob = policy.sample(
                        feats_t, temperature,
                        entropy_prior=entropies,
                        prior_alpha=prior_alpha,
                    )

                entropy_before = float(entropies[col])
                cand = apply_rl_mutation(current, col, op, rng)

            n_steps += 1

            if cand == current:
                stall += 1
                continue

            sp_old = sum_of_pairs(current)
            sp_new = sum_of_pairs(cand)
            dsp = sp_new - sp_old

            try:
                u_old = alignment_uncertainty(current, sequence_type=seq_type)
                u_new = alignment_uncertainty(cand, sequence_type=seq_type)
                du = u_old - u_new
            except Exception:
                du = 0.0

            # Behavioral signals (column-operator mode only)
            if policy_type == "column_op":
                step_alpha.append(abs(dsp) / max(1.0, abs(sp_old)))
                step_beta.append(entropy_before)
                step_gamma.append(_fast_gap_changed_fraction(current, cand))

            # Reward (Eq. 31)
            reward = (
                2.0 * dsp * (1.0 + 3.5 * entropy_before)
                + 4.0 * du
                + 2.0 * (1.0 if sp_new > best_sp else 0.0)
            )
            reward = max(-10.0, min(10.0, reward))

            if log_prob is not None:
                ep_log_probs.append(log_prob)
                ep_rewards.append(reward)

            # EO acceptance
            if sp_new >= sp_old:
                current = cand
                if sp_new > best_sp:
                    best_sp = sp_new
                    best = list(current)
                    stall = 0
                else:
                    stall += 1
            if stall >= stall_threshold:
                current = list(best)
                stall = 0

        # ---- Aggregate to episode level ----
        row = {"episode": ep,
               "phase": "explore" if ep < n_episodes // 2 else "exploit"}

        if policy_type == "weights" and step_true_weights:
            w = np.array(step_true_weights)
            row["alpha_eff"] = float(w[:, 0].mean())
            row["beta_eff"] = float(w[:, 1].mean())
            row["gamma_eff"] = float(w[:, 2].mean())
        else:
            row["alpha_eff"] = _mean_or_zero(step_alpha)
            row["beta_eff"] = _mean_or_zero(step_beta)
            row["gamma_eff"] = _mean_or_zero(step_gamma)

        s = row["alpha_eff"] + row["beta_eff"] + row["gamma_eff"] + 1e-12
        row["alpha_share"] = row["alpha_eff"] / s
        row["beta_share"] = row["beta_eff"] / s
        row["gamma_share"] = row["gamma_eff"] / s
        rows.append(row)

        # ---- Policy update ----
        if ep_log_probs and optimizer is not None:
            returns = np.zeros(len(ep_rewards))
            R = 0.0
            for k in range(len(ep_rewards) - 1, -1, -1):
                R = ep_rewards[k] + 0.95 * R
                returns[k] = R
            baseline = float(np.mean(returns))
            adv = torch.tensor(returns - baseline, dtype=torch.float32)
            loss = sum(-lp * a for lp, a in zip(ep_log_probs, adv))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        elif ep_log_probs and hasattr(policy, "update"):
            try:
                policy.update(ep_log_probs, ep_rewards)
            except Exception:
                pass

    return pd.DataFrame(rows), policy_type


# ---------------------------------------------------------------------------
# Table 28 builder
# ---------------------------------------------------------------------------
def build_table28(long, output_dir):
    phasewise = (
        long.groupby(["dataset", "seed", "phase"])
        [["alpha_eff", "beta_eff", "gamma_eff",
          "alpha_share", "beta_share", "gamma_share"]]
        .mean().reset_index()
    )
    phasewise.to_csv(output_dir / "phasewise_means.csv", index=False)

    rows = []
    for phase in ["explore", "exploit"]:
        sub = phasewise[phasewise["phase"] == phase]
        if sub.empty:
            continue
        row = {
            "Phase": phase.title(),
            "Mean α_eff":  round(sub["alpha_eff"].mean(), 6),
            "Mean β_eff":  round(sub["beta_eff"].mean(), 6),
            "Mean γ_eff":  round(sub["gamma_eff"].mean(), 6),
            "Normalized α_eff share": round(sub["alpha_share"].mean(), 6),
            "Normalized β_eff share": round(sub["beta_share"].mean(), 6),
            "Normalized γ_eff share": round(sub["gamma_share"].mean(), 6),
        }

        cv_values = []
        for ds in sub["dataset"].unique():
            d = sub[sub["dataset"] == ds]
            for col in ("alpha_eff", "beta_eff", "gamma_eff"):
                mu = d[col].mean()
                sd = d[col].std(ddof=0)
                if mu > 1e-9:
                    cv_values.append(sd / mu)
        row["CV across runs"] = round(float(np.mean(cv_values)), 6) if cv_values else 0.0
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "Table28.csv", index=False)

    # ---- markdown table (safe, no external dep) ----
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in table.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (int, float)):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    (output_dir / "Table28.md").write_text(
        "\n".join(lines), encoding="utf-8")

    return table


# ---------------------------------------------------------------------------
# Figure 9 builder
# ---------------------------------------------------------------------------
def build_figure9(long, output_dir, n_seeds):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    panels = [
        ("alpha_eff", "α_eff  (SP improvement signal)"),
        ("beta_eff",  "β_eff  (entropy of the selected column)"),
        ("gamma_eff", "γ_eff  (gap-change fraction)"),
    ]

    for ax, (col, label) in zip(axes, panels):
        for ds in long["dataset"].unique():
            d = long[long["dataset"] == ds]
            m = d.groupby("episode")[col].mean()
            ax.plot(m.index, m.values, lw=1.2, alpha=0.65, label=ds)
        n_ep = int(long["episode"].max())
        ax.axvline(n_ep / 2, color="black", ls="--", lw=1.2,
                   label="T_switch" if ax is axes[0] else None)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Episode")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(
        "Figure 9 — Effective reward signals across training\n"
        f"({n_seeds} seeds × {long['dataset'].nunique()} datasets)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "Figure9.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Load defaults, override with quick mode if requested ----
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    datasets = list(DEFAULT_DATASETS)
    seeds = list(DEFAULT_SEEDS)
    budget = DEFAULT_BUDGET

    if QUICK_MODE:
        datasets = datasets[:2]
        seeds = [42, 43, 44]
        budget = 500
        output_dir = "output_weights_quick"

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        print(f"\n[FATAL] input_dir not found: {input_path.resolve()}")
        print("Place this script next to the 'input' folder, or edit "
              "DEFAULT_INPUT_DIR at the top of the file.")
        return

    # ---- config ----
    cfg = make_cfg()
    cfg.MAFFT_CMD = base_cfg.MAFFT_CMD

    # ---- banner ----
    print("=" * 70)
    print("  Section 5.3.10 — Analysis of Effective Reward Signals")
    print("=" * 70)
    print(f"  input_dir : {input_path.resolve()}")
    print(f"  output_dir: {output_path.resolve()}")
    print(f"  datasets  : {datasets}")
    print(f"  seeds     : {seeds}")
    print(f"  budget    : {budget} mutations per (dataset, seed)")
    print(f"  policy    : "
          f"{'weight-learning (PGTrainer)' if _WEIGHT_POLICY_AVAILABLE else 'column-operator (ColumnOperatorPolicy)'}")
    if QUICK_MODE:
        print("  [QUICK MODE: reduced settings]")
    print("=" * 70)

    # ---- run ----
    all_dfs = []
    policy_type_seen = None
    t_start = time.time()

    for ds in datasets:
        fasta = input_path / f"{ds}.fasta"
        if not fasta.exists():
            print(f"  [SKIP] not found: {fasta}")
            continue
        for sd in seeds:
            print(f"  running {ds} / seed={sd} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                df, ptype = run_one(str(fasta), sd, cfg, budget)
                policy_type_seen = ptype
                df["dataset"] = ds
                df["seed"] = sd
                all_dfs.append(df)
                print(f"done ({time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()

    if not all_dfs:
        print("\n[FATAL] No results produced.")
        return

    long = pd.concat(all_dfs, ignore_index=True)
    long.to_csv(output_path / "per_episode_weights.csv", index=False)

    # ---- Table 28 ----
    table = build_table28(long, output_path)

    # ---- Figure 9 ----
    build_figure9(long, output_path, n_seeds=len(seeds))

    # ---- summary ----
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  Table 28 — Phase-wise effective reward signals")
    print("=" * 70)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"  policy used : {policy_type_seen}")
    if policy_type_seen == "column_op":
        print("  NOTE        : α_eff, β_eff, γ_eff are behavioral proxies, "
              "not learned weights.")
    print(f"  total time  : {elapsed:.1f}s")
    print("=" * 70)
    print("  Saved files:")
    for f in sorted(output_path.glob("*")):
        print(f"    {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] Unhandled exception:")
        traceback.print_exc()
