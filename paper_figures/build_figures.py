"""Regenerate the paper's figures from the experiment's JSON outputs.

Run from the repo root:
    python paper_figures/build_figures.py

Produces (in this directory):
    fig1_compute_accuracy.{png,svg}   compute-vs-accuracy Pareto, 8B + 3B panels
    fig2_architecture.{png,svg}       schematic of inserted-layer vs LoRA placement
    fig3_training_dynamics.{png,svg}  training curves for condition B (LoRA + GRPO)

The figures depend only on JSON files under ./results, ./results_final, and
./results3B. No model weights or tarballs are read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# Headline numbers (multi-seed JSON aggregates supersede the older Results.md
# numbers; per-condition GSM8K pass@1, GSM8K-Hard pass@1, train time in min,
# trainable parameter count in M).
# ──────────────────────────────────────────────────────────────────────────────

EIGHTB = {
    # condition: (gsm8k, gsm8k_std, hard, hard_std, train_min, trainable_M, marker_label)
    "A":    (0.131, None,   0.039, None,   0,      0,    "A: Baseline"),
    "B":    (0.597, None,   0.224, None,   1320,   168,  "B: LoRA + GRPO"),
    "C":    (0.503, None,   0.159, None,   1200,   436,  "C: Inserted + GRPO"),
    "D-2L": (0.594, None,   0.207, None,   22,     436,  "D: Inserted + SFT (2L)"),
    "D-4L": (0.589, 0.003,  0.204, 0.004,  25,     872,  "D: Inserted + SFT (4L)"),
    "E":    (0.542, 0.005,  0.184, 0.003,  15,     436,  "E: LoRA → SFT inserted"),
    "G":    (0.542, 0.010,  0.192, 0.003,  20,     436,  "G: LoRA distillation"),
}

THREEB = {
    "A":    (0.080, None,  0.029, None,   0,    0,    "A: Baseline"),
    "B":    (0.321, None,  0.093, None,   600,  97,   "B: LoRA + GRPO"),
    "D-2L": (0.325, 0.005, 0.091, 0.001,  18,   201,  "D: Inserted + SFT (2L)"),
    "D-4L": (0.356, 0.007, 0.099, 0.001,  21,   403,  "D: Inserted + SFT (4L)"),
    "G":    (0.297, 0.007, 0.076, 0.004,  15,   201,  "G: LoRA distillation"),
}


COND_COLOR = {
    "A":    "#888888",
    "B":    "#d62728",   # red — RL baseline
    "C":    "#ff7f0e",   # orange — RL inserted (negative result)
    "D-2L": "#1f77b4",   # blue — primary headline
    "D-4L": "#5fa8d3",   # light blue
    "E":    "#2ca02c",   # green
    "G":    "#9467bd",   # purple
}

COND_MARKER = {
    "A":    "o",
    "B":    "s",
    "C":    "X",
    "D-2L": "D",
    "D-4L": "D",
    "E":    "^",
    "G":    "P",
}


def _scatter_panel(ax, data, title, ylim=(0, 0.65), annotate_compute_gap=False):
    """Plot a single compute-vs-accuracy panel.

    All points use a uniform marker size; single-seed points are drawn with a
    hollow marker face (no fill, coloured edge) so they are visually distinct
    from the multi-seed points. Multi-seed points are drawn solid with their
    error bar.
    """
    UNIFORM_MARKER_SIZE = 90
    for cond, (gsm, gsm_std, _, _, mins, _, label) in data.items():
        # Use 0.5 minutes for the baseline (A) so it shows up on log axis
        x = max(mins, 0.5)
        color = COND_COLOR[cond]
        marker = COND_MARKER[cond]
        is_multiseed = gsm_std is not None
        if is_multiseed:
            ax.errorbar(
                x, gsm, yerr=gsm_std, fmt=marker, color=color,
                markersize=8, markeredgecolor=color, markeredgewidth=1.0,
                ecolor=color, elinewidth=1.2, capsize=3,
                label=label, zorder=3,
            )
        else:
            # Hollow marker indicates n=1 (single seed), so the absence of
            # an error bar is visually unmistakable.
            ax.scatter(
                x, gsm, marker=marker,
                facecolors="none", edgecolors=color, linewidths=1.6,
                s=UNIFORM_MARKER_SIZE, label=label, zorder=3,
            )
        # Annotate next to point
        offset_x, offset_y = 1.15, -0.012
        if cond == "A":
            offset_x, offset_y = 1.4, 0.012
        if cond in ("D-2L",):
            offset_y = 0.018
        if cond == "D-4L":
            offset_y = -0.030
        if cond == "G":
            offset_x, offset_y = 1.15, 0.012
        # Mark single-seed points with " (n=1)" superscript
        annot = cond + (" (n=1)" if not is_multiseed and cond != "A" else "")
        ax.annotate(
            annot, (x, gsm), xytext=(x * offset_x, gsm + offset_y),
            fontsize=8.5, color=color, weight="bold",
        )

    # Pareto frontier hint: connect baseline -> D-2L -> B
    pareto_pts = []
    if "A" in data:
        pareto_pts.append((max(data["A"][4], 0.5), data["A"][0]))
    if "D-2L" in data:
        pareto_pts.append((data["D-2L"][4], data["D-2L"][0]))
    if "D-4L" in data and data["D-4L"][0] >= (data.get("D-2L", (0,))[0] or 0):
        pareto_pts.append((data["D-4L"][4], data["D-4L"][0]))
    if "B" in data and data["B"][0] >= max(p[1] for p in pareto_pts):
        pareto_pts.append((data["B"][4], data["B"][0]))
    pareto_pts.sort()
    if len(pareto_pts) >= 2:
        xs, ys = zip(*pareto_pts)
        ax.plot(xs, ys, "--", color="#444", alpha=0.4, linewidth=1.0, zorder=1,
                label="Pareto frontier")

    # Headline annotation: horizontal arrow from B to D-2L showing the
    # compute reduction, drawn at B's accuracy level.
    if annotate_compute_gap and "B" in data and "D-2L" in data:
        b_x, b_y = data["B"][4], data["B"][0]
        d_x, d_y = data["D-2L"][4], data["D-2L"][0]
        arrow_y = (b_y + d_y) / 2 + 0.045  # slightly above the two points
        ax.annotate(
            "", xy=(d_x, arrow_y), xytext=(b_x, arrow_y),
            arrowprops=dict(arrowstyle="->", color="#222", lw=1.2,
                            shrinkA=2, shrinkB=2),
            zorder=4,
        )
        ratio = b_x / d_x
        gap_pts = (b_y - d_y) * 100
        gap_label = (f"~{ratio:.0f}× less compute,\n"
                     f"{abs(gap_pts):.1f} pt accuracy "
                     f"{'gap' if gap_pts > 0 else 'gain'}")
        # Place text above the arrow midpoint (geometric mean for log-x)
        mid_x = (b_x * d_x) ** 0.5
        ax.annotate(
            gap_label, xy=(mid_x, arrow_y),
            xytext=(mid_x, arrow_y + 0.04),
            ha="center", fontsize=8.5, color="#222", weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#888", alpha=0.9),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training wall-clock (minutes, log scale)")
    ax.set_ylabel("GSM8K pass@1")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylim(*ylim)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0.3, 3000)


def figure_1_compute_accuracy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    _scatter_panel(ax1, EIGHTB, "Llama-3.1 8B", ylim=(0.05, 0.72),
                   annotate_compute_gap=True)
    _scatter_panel(ax2, THREEB, "Llama-3.2 3B", ylim=(0.0, 0.45))

    # Single legend below both axes; supplement with hollow-marker note for n=1
    handles, labels = ax1.get_legend_handles_labels()
    seen = set()
    deduped = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            deduped.append((h, l))
    # Add a synthetic legend entry explaining the hollow-marker convention
    n1_marker = Line2D([0], [0], marker="o", linestyle="none",
                        markerfacecolor="none", markeredgecolor="#444",
                        markeredgewidth=1.6, markersize=8,
                        label="hollow marker = single seed (n=1)")
    deduped.append((n1_marker, n1_marker.get_label()))
    fig.legend(
        [h for h, _ in deduped], [l for _, l in deduped],
        loc="lower center", ncol=4, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.suptitle(
        "Figure 1: Compute–accuracy frontier on GSM8K. SFT into inserted layers (D-2L)\n"
        "matches LoRA + GRPO (B) at ~60× lower compute on 8B; on 3B it ties B at "
        "~30× less compute and a 4-layer variant overtakes B at 4× B's params.",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"fig1_compute_accuracy.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig1_compute_accuracy.{png,svg}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: architecture schematic (LoRA-style "width" delta vs inserted-layer
# "depth" delta on a frozen 32-layer base).
# ──────────────────────────────────────────────────────────────────────────────

def figure_2_architecture():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 6))

    def draw_stack(ax, title, subtitle, lora=False, inserted_at=None):
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 36)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, weight="bold")

        n_base = 32
        # Frozen layers
        for i in range(n_base):
            color = "#dcdcdc"
            ax.add_patch(mpatches.Rectangle(
                (3, 1 + i), 4, 0.9, facecolor=color, edgecolor="#666",
                linewidth=0.6,
            ))
            if lora:
                # Small adapter notch attached to the right of every layer
                ax.add_patch(mpatches.Rectangle(
                    (7.1, 1 + i + 0.15), 0.6, 0.6,
                    facecolor=COND_COLOR["B"], edgecolor=COND_COLOR["B"],
                ))
            # Layer-index labels every 8 layers, on the left, for spatial reference
            if i % 8 == 0 or i == n_base - 1:
                ax.text(2.7, 1 + i + 0.45, f"L{i}", fontsize=6.5,
                        color="#888", ha="right", va="center")
        # Inserted full layers (drawn between base layers if specified)
        if inserted_at:
            inserted_at = sorted(inserted_at)
            yoff = 0
            for i in range(n_base):
                if i in inserted_at:
                    yoff += 1.05
                    yi2 = 1 + i + yoff - 0.05
                    ax.add_patch(mpatches.Rectangle(
                        (3, yi2), 4, 0.95,
                        facecolor=COND_COLOR["D-2L"], edgecolor="black",
                        linewidth=1.0,
                    ))
                    # Zero-init badge: white circle with "0" centred in the
                    # block, indicating zero-initialised output projections
                    # → identity at step 0. Colour + legend already convey
                    # "inserted full layer", so no in-block text is needed.
                    ax.add_patch(mpatches.Circle(
                        (5.0, yi2 + 0.47), 0.34,
                        facecolor="white", edgecolor="black", linewidth=0.9,
                        zorder=5,
                    ))
                    ax.text(5.0, yi2 + 0.47, "0", fontsize=8,
                            ha="center", va="center", weight="bold",
                            color="black", zorder=6)
                    # Insertion-position label outside the block, on the right
                    ax.text(7.3, yi2 + 0.45, f"after L{i}", fontsize=7,
                            color=COND_COLOR["D-2L"], ha="left", va="center",
                            weight="bold")
            ax.set_ylim(0, 36 + yoff)

        # input/output callouts
        legend_x = 0.4
        ax.text(legend_x, ax.get_ylim()[1] - 0.5, "input ↓", fontsize=9)
        ax.text(legend_x, 0.4, "output ↑", fontsize=9)
        # Subtitle: trainable parameter count
        ax.text(5, -1.6, subtitle, ha="center", va="top",
                fontsize=10, weight="bold", color="#222")

    draw_stack(
        axL,
        "LoRA r=64 (B): rank-64 deltas\non every linear layer",
        subtitle="trainable params: ≈168 M",
        lora=True,
    )
    draw_stack(
        axR,
        "Inserted layers (C/D/E/G):\n2 full LlamaDecoderLayer blocks\n"
        "(0 = zero-init out-proj → identity at step 0)",
        subtitle="trainable params: ≈436 M  (~2.6× LoRA r=64)",
        inserted_at=[10, 21],
    )

    legend_handles = [
        mpatches.Patch(facecolor="#dcdcdc", edgecolor="#666", label="frozen base layer"),
        mpatches.Patch(facecolor=COND_COLOR["B"], label="LoRA adapter (rank-64)"),
        mpatches.Patch(facecolor=COND_COLOR["D-2L"], label="inserted full layer (trainable)"),
        mpatches.Patch(facecolor="white", edgecolor="black",
                       label="0 = zero-initialised output projection"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Figure 2: Two ways to add a fixed compute budget to a frozen LLM. "
        "LoRA applies dense low-rank deltas to every existing layer; inserted layers\n"
        "add sparse full-rank blocks at chosen positions ([L10, L21] here). "
        "Both freeze the base. Compute-matched, but inserted recipes carry ~2.6× more trainable params.",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"fig2_architecture.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig2_architecture.{png,svg}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: training dynamics. The GRPO panels overlay condition B (LoRA + GRPO)
# against condition C (inserted layers + GRPO) for reward, KL, and gradient norm
# — directly substantiating §5.2's structural claims about why RL through
# inserted layers underperforms. The fourth panel shows condition D's SFT loss
# and token accuracy (Appendix E). Curves are parsed from the archived wandb
# output.log files by extract_training_curves.py into training_curves.json.
#
# Condition G (distillation) used a custom AdamW loop not logged to this wandb
# project; its KL-loss curve is not recoverable and remains prose-described.
# ──────────────────────────────────────────────────────────────────────────────

def _finite_xy(steps, vals):
    """Drop (step, value) pairs where value is missing/non-finite."""
    xs, ys = [], []
    for x, y in zip(steps, vals):
        if isinstance(y, (int, float)):
            xs.append(x)
            ys.append(y)
    return xs, ys


def _moving_avg(ys, window=5):
    """Centred moving average for readability over noisy step-level metrics."""
    if len(ys) < window:
        return ys
    out = []
    half = window // 2
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        seg = ys[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def figure_3_training_dynamics():
    curves_path = FIG_DIR / "training_curves.json"
    if not curves_path.exists():
        raise SystemExit(
            "training_curves.json missing — run "
            "`python paper_figures/extract_training_curves.py` first."
        )
    conds = json.loads(curves_path.read_text())["conditions"]
    B, C, D = conds.get("B"), conds.get("C"), conds.get("D")

    fig, (ax_r, ax_k, ax_g, ax_d) = plt.subplots(1, 4, figsize=(17.5, 4))

    def overlay(ax, key, label):
        """Plot B vs C for one GRPO metric: raw (faint) + moving average."""
        for cond, name in ((B, "B: LoRA + GRPO"), (C, "C: Inserted + GRPO")):
            if not cond or key not in cond:
                continue
            color = COND_COLOR["B"] if cond is B else COND_COLOR["C"]
            xs, ys = _finite_xy(cond["step"], cond[key])
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25)
            ax.plot(xs, _moving_avg(ys), color=color, linewidth=1.8, label=name)
        ax.set_xlabel("Training step")
        ax.grid(True, which="both", alpha=0.3)

    # (a) Reward — the headline contrast: B climbs to ~0.75, C plateaus ~0.5
    overlay(ax_r, "reward", "reward")
    ax_r.set_ylabel("Mean reward (per group)")
    ax_r.set_title("Reward: B vs C (GRPO)", fontsize=11, weight="bold")
    ax_r.set_ylim(0, 0.85)
    ax_r.legend(fontsize=8, loc="lower right")

    # (b) KL to reference — C drifts higher and noisier than B
    overlay(ax_k, "kl", "kl")
    ax_k.set_ylabel("KL(π ∥ π₀)")
    ax_k.set_title("KL to reference: B vs C", fontsize=11, weight="bold")
    ax_k.legend(fontsize=8, loc="upper left")

    # (c) Gradient norm (log) — C runs ~10–40 vs B's ~0.5–1.0
    overlay(ax_g, "grad_norm", "grad_norm")
    ax_g.set_yscale("log")
    ax_g.set_ylabel("Gradient norm (log)")
    ax_g.set_title("Gradient norm: B vs C", fontsize=11, weight="bold")
    ax_g.legend(fontsize=8, loc="upper right")

    # (d) Condition D SFT dynamics — CE loss (left) + token accuracy (right)
    if D:
        xs, ys = _finite_xy(D["step"], D.get("loss", []))
        ax_d.plot(xs, ys, color=COND_COLOR["D-2L"], linewidth=0.8, alpha=0.25)
        ax_d.plot(xs, _moving_avg(ys), color=COND_COLOR["D-2L"], linewidth=1.8,
                  label="CE loss")
        ax_d.set_ylabel("Cross-entropy loss", color=COND_COLOR["D-2L"])
        ax_d.tick_params(axis="y", labelcolor=COND_COLOR["D-2L"])
        if "mean_token_accuracy" in D:
            ax_t = ax_d.twinx()
            xs2, ys2 = _finite_xy(D["step"], D["mean_token_accuracy"])
            ax_t.plot(xs2, _moving_avg(ys2), color=COND_COLOR["D-4L"],
                      linewidth=1.8, linestyle="--", label="token accuracy")
            ax_t.set_ylabel("Token accuracy", color=COND_COLOR["D-4L"])
            ax_t.tick_params(axis="y", labelcolor=COND_COLOR["D-4L"])
            ax_t.set_ylim(0.5, 0.9)
    ax_d.set_xlabel("Training step")
    ax_d.set_title("D: Inserted + SFT dynamics", fontsize=11, weight="bold")
    ax_d.grid(True, alpha=0.3)

    fig.suptitle(
        "Figure 3: Training dynamics on Llama-3.1 8B, reconstructed from the archived wandb logs. "
        "Panels (a–c) overlay LoRA + GRPO (B) against inserted-layer GRPO (C): C's reward plateaus near "
        "0.5 vs B's ~0.75,\nits KL drifts higher, and its gradient norms run ~10–40 vs B's ~0.5–1.0 — the "
        "instability §5.2 describes. Panel (d): condition D's SFT loss falls 1.76→~0.7 as token accuracy "
        "rises to ~0.82.\nCondition G (custom distillation loop) was not logged to this project and is "
        "omitted.",
        fontsize=9.5, y=1.12,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"fig3_training_dynamics.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig3_training_dynamics.{png,svg}")


def main():
    figure_1_compute_accuracy()
    figure_2_architecture()
    figure_3_training_dynamics()
    print("All figures regenerated in", FIG_DIR)


if __name__ == "__main__":
    main()
