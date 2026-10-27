# main.py

import config as cfg
import numpy as np
import random
import torch

from fasta_io import read_fasta
from mafft_seed import run_mafft
from scoring import sum_of_pairs
from uncertainty import alignment_uncertainty
from eo_optimizer import eo_optimize
from pg_trainer import PGTrainer

from plot_rl_curve import plot_rl_learning
from column_confidence import (
    compute_column_confidence,
    plot_guidance_style
)

# --------------------------------------------------
# Settings
# --------------------------------------------------
N_RUNS = 15
N_EPISODES = 60

# --------------------------------------------------
# Header
# --------------------------------------------------
print("=" * 60)
print(" RL-guided Uncertainty-Aware MSA (Policy Gradient) ")
print("=" * 60)

# --------------------------------------------------
# Load sequences (once)
# --------------------------------------------------
seqs = read_fasta("input.fasta")

all_results = []

# ==================================================
# Multiple runs
# ==================================================
for run_id in range(N_RUNS):

    print("\n" + "=" * 60)
    print(f" RUN {run_id + 1} / {N_RUNS}")
    print("=" * 60)

    # --- reproducibility (controlled randomness) ---
    seed = 1000 + run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------
    # 1. MAFFT seed alignment
    # --------------------------------------------------
    print("\n[1] Running MAFFT seed alignment...")
    seed_alignment = run_mafft(seqs, cfg.MAFFT_CMD)

    seed_sp  = sum_of_pairs(seed_alignment)
    seed_unc = alignment_uncertainty(seed_alignment)

    print(f"Seed SP          : {seed_sp}")
    print(f"Seed Uncertainty : {seed_unc:.4f}")

    # --------------------------------------------------
    # 2. RL training
    # --------------------------------------------------
    trainer = PGTrainer(cfg)

    print("\n[2] RL Training]")
    for ep in range(N_EPISODES):
        info = trainer.train_step(seed_alignment)

        print(
            f" Ep {ep:02d} | "
            f"R={info['reward']:.3f} | "
            f"α={info['alpha']:.2f}, "
            f"β={info['beta']:.2f}, "
            f"γ={info['gamma']:.2f}"
        )

    # --------------------------------------------------
    # 3. Final EO optimization
    # --------------------------------------------------
    print("\n[3] Final Optimization with learned parameters]")

    final_alignment, _ = eo_optimize(
        seed_alignment,
        cfg
    )

    final_sp  = sum_of_pairs(final_alignment)
    final_unc = alignment_uncertainty(final_alignment)

    print(f"Final SP         : {final_sp}")
    print(f"Final Uncertainty: {final_unc:.4f}")

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------
    all_results.append({
        "seed_sp": seed_sp,
        "final_sp": final_sp,
        "seed_unc": seed_unc,
        "final_unc": final_unc,
        "alpha": cfg.ALPHA,
        "beta": cfg.BETA,
        "gamma": cfg.GAMMA,
        "alignment": final_alignment,
        "seed_alignment": seed_alignment,
        "history": trainer.history
    })

# ==================================================
# Aggregate results
# ==================================================
final_sps  = np.array([r["final_sp"] for r in all_results])
final_uncs = np.array([r["final_unc"] for r in all_results])

mean_sp  = final_sps.mean()
std_sp   = final_sps.std()

mean_unc = final_uncs.mean()
std_unc  = final_uncs.std()

# Best run: max SP, tie → min uncertainty
best_idx = np.lexsort((final_uncs, -final_sps))[0]
best = all_results[best_idx]

# ==================================================
# Final report
# ==================================================
print("\n" + "=" * 60)
print(" FINAL MULTI-RUN REPORT ")
print("=" * 60)

print(f"Total Runs            : {N_RUNS}")
print(f"Mean Final SP         : {mean_sp:.2f} ± {std_sp:.2f}")
print(f"Mean Final Uncertainty: {mean_unc:.4f} ± {std_unc:.4f}")

print("\n--- BEST RUN ---")
print(f"Run Index             : {best_idx + 1}")
print(f"Initial SP            : {best['seed_sp']}")
print(f"Final SP              : {best['final_sp']}")
print(f"SP Improvement        : {best['final_sp'] - best['seed_sp']}")

print(f"\nInitial Uncertainty   : {best['seed_unc']:.4f}")
print(f"Final Uncertainty     : {best['final_unc']:.4f}")

print(
    f"\nFinal α, β, γ         : "
    f"{best['alpha']:.2f}, {best['beta']:.2f}, {best['gamma']:.2f}"
)

print("\n=== Best Alignment ===")
for s in best["alignment"]:
    print(s)

# --------------------------------------------------
# GUIDANCE-style column confidence (best run only)
# --------------------------------------------------
print("\n[4] GUIDANCE-style column confidence analysis]")

conf_mafft = compute_column_confidence(best["seed_alignment"], sequence_type=getattr(cfg, "SEQUENCE_TYPE", "auto"),)
conf_rl    = compute_column_confidence(best["alignment"], sequence_type=getattr(cfg, "SEQUENCE_TYPE", "auto"),)

plot_guidance_style(
    conf_rl=conf_rl,
    conf_mafft=conf_mafft,
    threshold=0.6
)

print("\nPipeline completed successfully ✔")
