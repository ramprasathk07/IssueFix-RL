"""Render training-run charts for README/blog from exported wandb history.

Inputs : wandb_history_jaml5ohf.json (final run), wandb_history_ug94ujsd.json (first run)
Outputs: assets/*.png (light-mode, 2x scale)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── palette (light mode) ────────────────────────────────────────────────────
SURFACE   = "#fcfcfb"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
BLUE      = "#2a78d6"   # series 1
AQUA      = "#1baf7a"   # series 2 (sub-3:1 on light -> direct labels)
YELLOW    = "#eda100"   # series 3
RED       = "#e34948"   # series 6

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "Segoe UI",
    "text.color": INK_2,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "600",
    "axes.labelsize": 10,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_2,
})

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent


def load(run_id):
    return json.load(open(ROOT / "assets" / "data" / f"wandb_history_{run_id}.json"))


def series(hist, key, x="_step"):
    pts = [(h[x], h[key]) for h in hist if h.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def epoch_lines(ax, steps=(63, 126)):
    for s in steps:
        ax.axvline(s, color=GRID, linewidth=1.2, linestyle=(0, (4, 3)), zorder=0)


final = load("jaml5ohf")
first = load("ug94ujsd")

# ── 1. training dynamics 2x2 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=200)
fig.subplots_adjust(hspace=0.42, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.08)
fig.suptitle("SFT run qwen0.5_sft_2gpu — training dynamics (3 epochs, 189 optimizer steps)",
             fontsize=13, fontweight="600", color=INK, x=0.07, ha="left")

# loss
ax = axes[0][0]
xs, ys = series(final, "train/loss")
vx, vy = series(final, "val/loss")
ax.plot(xs, ys, color=BLUE, linewidth=2, label="train")
ax.plot(vx, vy, color=AQUA, linewidth=2, marker="o", markersize=7,
        markerfacecolor=AQUA, markeredgecolor=SURFACE, markeredgewidth=2, label="val")
for x_, y_ in zip(vx, vy):
    ax.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8.5, color=INK_2)
epoch_lines(ax)
ax.set_title("DFT loss")
ax.set_xlabel("optimizer step")
ax.legend(loc="upper right")

# entropy
ax = axes[0][1]
xs, ys = series(final, "train/entropy")
ax.plot(xs, ys, color=BLUE, linewidth=2)
ax.annotate(f"{ys[0]:.2f}", (xs[0], ys[0]), textcoords="offset points", xytext=(8, 0),
            fontsize=8.5, color=INK_2)
ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=8.5, color=INK_2)
epoch_lines(ax)
ax.set_title("Token entropy (nats)")
ax.set_xlabel("optimizer step")

# lr
ax = axes[1][0]
xs, ys = series(final, "train/lr")
ax.plot(xs, [y * 1e5 for y in ys], color=BLUE, linewidth=2, marker="o", markersize=5,
        markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5)
epoch_lines(ax)
ax.set_title("Learning rate (×1e-5) — note the rebound")
ax.set_xlabel("optimizer step")

# grad norm
ax = axes[1][1]
xs, ys = series(final, "train/grad_norm")
ax.plot(xs, ys, color=BLUE, linewidth=2)
epoch_lines(ax)
ax.set_title("Gradient norm (post-clip)")
ax.set_xlabel("optimizer step")

fig.savefig(ASSETS / "training_dynamics.png")
plt.close(fig)

# ── 2. LR schedule: intended vs actual ──────────────────────────────────────
WARMUP, TOTAL, PEAK = 50, 189, 2e-5

def cosine_lr(step, speed=1.0):
    s = step * speed
    if s < WARMUP:
        return PEAK * s / WARMUP
    progress = (s - WARMUP) / (TOTAL - WARMUP)
    return PEAK * max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

steps = list(range(0, TOTAL + 1))
intended = [cosine_lr(s) * 1e5 for s in steps]
effective = [cosine_lr(s, speed=2.0) * 1e5 for s in steps]  # scheduler stepped 2x per optimizer step
ax_, ay_ = series(final, "train/lr")

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.13)
ax.plot(steps, intended, color=MUTED, linewidth=2, linestyle=(0, (5, 4)), label="intended (cosine, 189 steps)")
ax.plot(steps, effective, color=BLUE, linewidth=2, label="what actually ran (scheduler stepped 2× in DDP)")
ax.plot(ax_, [y * 1e5 for y in ay_], color=BLUE, linewidth=0, marker="o", markersize=7,
        markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=2, label="wandb-logged values")
epoch_lines(ax)
ax.set_title("The cosine schedule that ran twice as fast — and came back up", loc="left")
ax.set_xlabel("optimizer step")
ax.set_ylabel("learning rate (×1e-5)")
ax.legend(loc="lower left")
ax.annotate("LR ≈ 0 at mid-training", (97, 0.62), ha="center", fontsize=9, color=INK_2)
ax.annotate("back to ~max by epoch 3", (105, 1.82), ha="center", fontsize=9, color=INK_2,
            xytext=(105, 1.82), textcoords="data")
ax.annotate("", xy=(152, 1.94), xytext=(122, 1.82),
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1.2))
fig.savefig(ASSETS / "lr_schedule.png")
plt.close(fig)

# ── 4. behavioral eval: base vs finetuned ───────────────────────────────────
# numbers from docs/eval/*.txt (eval_checkpoint.py, greedy, max_new_tokens=512)
metrics    = ["Runs correctly\n(exec pass)", "Stops before\n512-token cap", "Uses <answer>\ntags"]
base_pct   = [7 / 8 * 100, 8 / 8 * 100, 2 / 8 * 100]
ft_pct     = [4 / 7 * 100, 0.0, 0.0]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200, width_ratios=[2.1, 1])
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14, wspace=0.25)
fig.suptitle("Behavioral eval, 8 coding tasks — the finetune made the model worse",
             fontsize=13, fontweight="600", color=INK, x=0.06, ha="left")

w = 0.34
xpos = range(len(metrics))
b1 = ax.bar([x - w / 2 - 0.01 for x in xpos], base_pct, width=w, color=BLUE,
            label="base Qwen2.5-0.5B-Instruct", zorder=3)
b2 = ax.bar([x + w / 2 + 0.01 for x in xpos], ft_pct, width=w, color=AQUA,
            label="finetuned checkpoint", zorder=3)
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}%", (r.get_x() + r.get_width() / 2, r.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=INK_2)
ax.set_xticks(list(xpos))
ax.set_xticklabels(metrics, color=INK_2)
ax.set_ylim(0, 118)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("% of tasks")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right")

toks = [112.1, 512.0]
bars = ax2.bar([0, 1], toks, width=0.55, color=[BLUE, AQUA], zorder=3)
for r, v, note in zip(bars, toks, ["", " (hit cap 8/8)"]):
    ax2.annotate(f"{v:.0f}{note}", (r.get_x() + r.get_width() / 2, v),
                 textcoords="offset points", xytext=(0, 4), ha="center",
                 fontsize=9, color=INK_2)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["base", "finetuned"], color=INK_2)
ax2.set_ylim(0, 590)
ax2.set_title("Avg tokens per response")
ax2.grid(axis="x", visible=False)

fig.savefig(ASSETS / "eval_comparison.png")
plt.close(fig)

# ── 3. run comparison: first attempt vs final run ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.13)

fx = [h["_runtime"] / 3600 for h in first if h.get("train/loss") is not None]
fy = [h["train/loss"] for h in first if h.get("train/loss") is not None]
gx = [h["_runtime"] / 3600 for h in final if h.get("train/loss") is not None]
gy = [h["train/loss"] for h in final if h.get("train/loss") is not None]

ax.plot(fx, fy, color=YELLOW, linewidth=2, label="first attempt — 1 GPU, effective batch 2 (crashed)")
ax.plot(gx, gy, color=BLUE, linewidth=2, label="final run — 2× T4 DDP, effective batch 64")
ax.annotate(f"stuck ≈ {fy[-1]:.3f}\nafter {fx[-1]:.1f} h", (fx[-1], fy[-1]),
            textcoords="offset points", xytext=(8, 6), fontsize=9, color=INK_2)
ax.annotate(f"{gy[-1]:.3f}", (gx[-1], gy[-1]), textcoords="offset points", xytext=(6, 4),
            fontsize=9, color=INK_2)
ax.set_title("Same model, same data — what effective batch size and DDP bought", loc="left")
ax.set_xlabel("wall-clock hours")
ax.set_ylabel("train DFT loss")
ax.legend(loc="upper right")
fig.savefig(ASSETS / "run_comparison.png")
plt.close(fig)

print("wrote:", *[p.name for p in sorted(ASSETS.glob("*.png"))])
