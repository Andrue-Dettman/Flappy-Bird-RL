"""
reads training logs and generates plots
usage: python plot_training.py
"""

import os
import csv
import matplotlib.pyplot as plt

os.makedirs("output/figures", exist_ok=True)


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def smooth(vals, window=10):
    """simple moving average"""
    out = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        out.append(sum(vals[start:i+1]) / (i - start + 1))
    return out


# --- neuroevo plots ---
rows = read_csv("logs/neuroevo.csv")
if rows:
    gens = [int(r["gen"]) for r in rows]
    scores = [int(r["score"]) for r in rows]
    best = [int(r["best_ever"]) for r in rows]
    mean_fit = [float(r["mean_fit"]) for r in rows]
    stages = [int(r["stage"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Neuroevolution Training", fontsize=13)

    axes[0].plot(gens, scores, alpha=0.4, color="steelblue", label="gen score")
    axes[0].plot(gens, best, color="steelblue", linewidth=2, label="best ever")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("score (pipes passed)")
    axes[0].legend()
    axes[0].set_title("Score over Generations")

    # shade curriculum stages differently
    colors = ["#eaf4fb", "#d0e8f5", "#aed6ef", "#86bee7", "#5ea6df"]
    prev_stage, prev_gen = stages[0], gens[0]
    for i in range(1, len(stages)):
        if stages[i] != prev_stage or i == len(stages) - 1:
            axes[1].axvspan(prev_gen, gens[i], alpha=0.4,
                            color=colors[min(prev_stage, len(colors)-1)],
                            label=f"stage {prev_stage}")
            prev_stage, prev_gen = stages[i], gens[i]

    axes[1].plot(gens, mean_fit, color="coral", linewidth=1.5)
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("mean population fitness")
    axes[1].set_title("Fitness + Curriculum Stages")

    plt.tight_layout()
    plt.savefig("output/figures/neuroevo.png", dpi=150)
    plt.close()
    print("saved output/figures/neuroevo.png")
else:
    print("no neuroevo log found — run train.py first")


# --- dqn plots ---
rows = read_csv("logs/dqn.csv")
if rows:
    eps = [int(r["episode"]) for r in rows]
    scores = [int(r["score"]) for r in rows]
    rewards = [float(r["reward"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    epsilon = [float(r["epsilon"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DQN Training", fontsize=13)

    axes[0].plot(eps, scores, alpha=0.3, color="mediumseagreen", label="score")
    axes[0].plot(eps, smooth(scores, 30), color="mediumseagreen",
                 linewidth=2, label="smoothed")
    ax2 = axes[0].twinx()
    ax2.plot(eps, epsilon, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_ylabel("epsilon", color="gray")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("score (pipes passed)")
    axes[0].set_title("Score + Epsilon Decay")
    axes[0].legend(loc="upper left")

    # filter out zero-loss episodes (before warmup)
    loss_eps = [eps[i] for i in range(len(losses)) if losses[i] > 0]
    loss_vals = [losses[i] for i in range(len(losses)) if losses[i] > 0]
    if loss_vals:
        axes[1].plot(loss_eps, loss_vals, alpha=0.3, color="tomato")
        axes[1].plot(loss_eps, smooth(loss_vals, 30), color="tomato", linewidth=2)
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Training Loss")

    plt.tight_layout()
    plt.savefig("output/figures/dqn.png", dpi=150)
    plt.close()
    print("saved output/figures/dqn.png")
else:
    print("no dqn log found — run train_dqn.py first")
