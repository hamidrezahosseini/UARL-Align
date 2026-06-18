# plot_rl_curve.py
import matplotlib.pyplot as plt


def plot_rl_learning(history, mafft_sp, mafft_unc):
    ep = range(len(history))
    r = [h["reward"] for h in history]
    a = [h["alpha"] for h in history]
    b = [h["beta"] for h in history]
    g = [h["gamma"] for h in history]

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(ep, r, label="Reward")
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ep, a, label="α")
    plt.plot(ep, b, label="β")
    plt.plot(ep, g, label="λ")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig("rl_learning_curve.png", dpi=300)
    plt.show()
