
import numpy as np
import matplotlib.pyplot as plt

from uncertainty import column_uncertainties


def compute_column_confidence(alignment, sequence_type="auto"):
    """
    Confidence is the complement of normalized column entropy:
        confidence_j = 1 - uncertainty_j
    """
    uncertainties = column_uncertainties(
        alignment,
        sequence_type=sequence_type,
    )
    return 1.0 - np.asarray(uncertainties, dtype=float)


def plot_guidance_style(conf_rl, conf_mafft, threshold=0.6):
    if len(conf_rl) != len(conf_mafft):
        raise ValueError(
            "The two confidence arrays must have equal lengths to plot "
            "them on the same column axis."
        )

    x = np.arange(len(conf_rl))

    plt.figure(figsize=(12, 4))
    plt.plot(x, conf_mafft, label="MAFFT", alpha=0.6)
    plt.plot(x, conf_rl, label="RL-Optimized", linewidth=2)
    plt.axhline(
        threshold,
        linestyle="--",
        label="Confidence Threshold",
    )
    plt.xlabel("Alignment Column")
    plt.ylabel("Confidence Score")
    plt.title("Column Confidence Based on Normalized Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("column_confidence_guidance.png", dpi=300)
    plt.show()
