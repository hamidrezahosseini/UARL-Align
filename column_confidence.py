# column_confidence.py

import numpy as np
import math
import matplotlib.pyplot as plt

def column_entropy(col):
    counts = {}
    for c in col:
        if c != '-':
            counts[c] = counts.get(c, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for v in counts.values():
        p = v / total
        entropy -= p * math.log(p + 1e-12)

    return entropy


def compute_column_confidence(alignment):
    """
    alignment: list of aligned sequences (strings)
    """
    n_cols = len(alignment[0])
    alphabet_size = len(
        set("".join(seq.replace("-", "") for seq in alignment))
    )
    H_max = math.log(alphabet_size + 1e-12)

    confidences = []

    for j in range(n_cols):
        col = [seq[j] for seq in alignment]
        H = column_entropy(col)
        conf = 1.0 - (H / H_max if H_max > 0 else 0)
        confidences.append(conf)

    return np.array(confidences)


def plot_guidance_style(conf_rl, conf_mafft, threshold=0.6):
    x = np.arange(len(conf_rl))

    plt.figure(figsize=(12, 4))

    plt.plot(x, conf_mafft, label="MAFFT", alpha=0.6)
    plt.plot(x, conf_rl, label="RL-Optimized", linewidth=2)

    plt.axhline(threshold, linestyle="--", label="Confidence Threshold")

    plt.xlabel("Alignment Column")
    plt.ylabel("Confidence Score")
    plt.title("GUIDANCE-style Column Confidence")
    plt.legend()

    plt.tight_layout()
    plt.savefig("column_confidence_guidance.png", dpi=300)
    plt.show()
