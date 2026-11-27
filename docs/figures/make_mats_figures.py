"""Rebuild every figure in docs/mats-writeup.md from the artifact CSVs in docs/figures/data/.

Run from the repo root:  python docs/figures/make_mats_figures.py

Every plotted number is read from copied run artifacts (final calibrated run, the 8B steering
confirmation run, the first scale run, the shortcut pilot re-analysis, probe layer profiles),
plus files computed afterwards from existing outputs (no new model runs):
  data/stage35_ifeval_truncation_check.csv   8B IFEval, items where no arm truncated
  data/stage35_ifeval_random_vs_nosteer.csv  paired bootstrap CI recomputed from predictions
  data/bigram_baseline.csv                   docs/figures/bigram_baseline.py on the probe stimuli
  data/cue_crosschecks.csv                   every calm/desperate cue vs neutral, every model
  data/changed_examples_seed2026.json        seeded samples of flipped items + flip counts

Palette: dataviz reference palette, validated (adjacent CVD dE >= 9.1; first three slots pass
all-pairs; slots 3/4 below 3:1 contrast, so every coloured series has a legend entry plus a
distinct marker or a direct label). Encoding rules used in the numbered figures
(the overview uses directly labelled estimates without significance encoding):
  steering vectors: diligent (+) BLUE square, careless (-) ORANGE circle, random AQUA triangle,
                    shuffled-label YELLOW diamond; no steering / reference GREY
  filled = survives Holm in its preregistered family; half-filled = exploratory, uncorrected
  p < .05; open = not significant; grey x = excluded by the coherence check.
Diverging grids use the blue <-> red pair with a grey midpoint.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE
# Keep existing asset filenames stable. Captions in the write-up supply the printed numbers.

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, MID = "#e34948", "#f0efec"
GREY = "#a8a79f"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
VEC = {"positive": ("diligent (+)", BLUE, "s"), "negative": ("careless (−)", ORANGE, "o"),
       "random": ("random", AQUA, "^"), "shuffled": ("shuffled-label", YELLOW, "D")}
W = 7.6

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.6,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "savefig.dpi": 240, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
ANN = 8.6


def rows(name: str) -> list[dict]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def pp(x) -> float:
    return 100 * float(x)


def fmt(v: float, nd: int = 1) -> str:
    return f"{v:+.{nd}f}".replace("-", "−")




SIZES = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]
LAB = {"Qwen/Qwen3-8B": "8B", "Qwen/Qwen3-14B": "14B", "Qwen/Qwen3-32B": "32B",
       "us.anthropic.claude-sonnet-5": "Sonnet 5"}
XS = {m: math.log10(s) for m, s in zip(SIZES, (8, 14, 32))}


def mark(ax, x, y, color, marker, status, size=46, z=3):
    """status: 'holm' filled, 'explore' half-filled, 'ns' open, 'excluded' grey x."""
    if status == "excluded":
        ax.scatter([x], [y], marker="X", s=size * 1.3, color=MUTED, edgecolor=SURFACE, linewidth=1.2, zorder=z)
    elif status == "explore":
        ax.plot([x], [y], marker=marker, ms=math.sqrt(size) * 1.05, color=color, markeredgecolor=color,
                markerfacecoloralt=SURFACE, fillstyle="left", lw=0, zorder=z)
    else:
        ax.scatter([x], [y], marker=marker, s=size, zorder=z, linewidth=1.6,
                   color=color if status == "holm" else SURFACE, edgecolor=SURFACE if status == "holm" else color)


def point(ax, x, y, lo, hi, color, marker, status, label=None, side="right", dx=0.0, size=46):
    x = x + dx
    ax.plot([x, x], [lo, hi], color=MUTED if status == "excluded" else color, lw=1.4, alpha=0.75, zorder=2)
    mark(ax, x, y, color, marker, status, size)
    if label:
        ax.annotate(label, (x, y), xytext=(7 if side == "right" else -7, 0), textcoords="offset points",
                    ha="left" if side == "right" else "right", va="center", fontsize=ANN, color=INK)


def size_axis(ax):
    ax.set_xticks(list(XS.values()), [LAB[m] for m in SIZES])
    ax.set_xlim(math.log10(8) - 0.17, math.log10(32) + 0.2)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=1)


STATUS_HANDLES = [
    Line2D([], [], marker="o", ls="", color=INK2, markersize=6, label="survives Holm"),
    Line2D([], [], marker="o", ls="", color=INK2, markerfacecoloralt=SURFACE, fillstyle="left", markersize=6,
           label="exploratory, uncorrected p < .05"),
    Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=6, label="not significant"),
    Line2D([], [], marker="X", ls="", color=MUTED, markersize=7, label="excluded (coherence)"),
]


# =================================================================================
# Figure 1 (overview): three controls that change the interpretation, for the top of the write-up
# =================================================================================
def fig0_overview() -> None:
    """Selected evidence, not a pooled estimate across models or experiments."""
    paired = {(r["condition"], r["baseline"]): r
              for r in rows("stage35_ifeval_paired_deltas.csv")
              + rows("stage35_ifeval_random_vs_nosteer.csv")
              if r["task"] == "ifeval" and r["signal"] == "strict_prompt"}
    diligent = paired[("positive_a+0.5000", "no_steer")]
    random = paired[("random_a+0.5000", "no_steer")]
    desperate = {name: next(r for r in rows(name) if r["model"] == SIZES[0]
                           and r["condition"] == "stage4_desperate_medium__none")
                 for name in ("safety_primary.csv", "overrefusal_monitoring.csv")}
    calm = next(r for r in rows("safety_primary.csv") if r["model"] == SIZES[2]
                and r["condition"] == "stage4_calm_high__none")
    neutral_delta = next(float(r["delta_pp"]) for r in rows("cue_crosschecks.csv")
                         if r["model"] == "32b" and r["cue"] == "neutral_vs_no_cue"
                         and r["outcome"] == "agreement_with_false_claims")
    neutral_rate = next(pp(r["held_out_rate"]) for r in rows("base_rates.csv")
                        if r["model"] == SIZES[2] and r["signal"] == "sycophancy")
    calm_delta = pp(calm["delta_vs_no_steer"])
    rates = [neutral_rate - neutral_delta, neutral_rate, neutral_rate + calm_delta]

    # Every 8B refusal answer in these two conditions parsed as REFUSE or COMPLY.
    # Express the harmful-request contrast as REFUSE here so both rows read the same way.
    refusal_rates = [r for r in rows("coherence_wall.csv") if r["model"] == SIZES[0]
                     and r["task"] == "refusal_drift" and r["condition"] in
                     ("neutral_length_matched__none", "stage4_desperate_medium__none")]
    if len(refusal_rates) != 2 or any(float(r["parse_success_mean"]) != 1 for r in refusal_rates):
        raise ValueError("Harmful compliance can only be inverted with complete binary responses")
    harmful = desperate["safety_primary.csv"]
    harmful_refusal = {"delta": -float(harmful["delta_vs_no_steer"]),
                       "ci_low": -float(harmful["ci_high"]), "ci_high": -float(harmful["ci_low"])}

    with plt.rc_context({"font.size": 11, "xtick.labelsize": 10,
                         "axes.labelsize": 10.5, "svg.fonttype": "none",
                         "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(10.5, 3.1))
        fig.text(.035, .96, "Cheap controls explain the emotional effects in Qwen Models",
                 fontsize=16.5, fontweight="bold", color=INK, va="top")
        starts = [.035, .365, .695]
        titles = ["1a   Instruction following", "1b   Refusal after an urgent cue",
                  "1c   Agreement with false claims"]
        for x, title in zip(starts, titles):
            fig.text(x, .79, title, fontsize=12.5, fontweight="bold", color=INK, va="top")

        def forest(x, xlim, ticks, items, xlabel):
            ax = fig.add_axes([x, .22, .280, .45])
            ax.set(xlim=xlim, ylim=(-.48, 1.55), yticks=[], xticks=ticks)
            ax.axvline(0, color=AXIS, lw=1.1)
            ax.spines["left"].set_visible(False)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(xlabel, labelpad=7)
            for y, (label, r, field, color, marker) in zip((1, 0), items):
                value, lo, hi = pp(r[field]), pp(r["ci_low"]), pp(r["ci_high"])
                ax.text(xlim[0], y + .28, label, fontsize=10.5, color=INK)
                ax.plot([lo, hi], [y, y], color=color, lw=2)
                ax.plot([value], [y], marker=marker, ms=7, color=color, lw=0)
                ax.text(xlim[1], y + .28, fmt(value), ha="right", fontsize=11,
                        fontweight="bold", color=INK)
            return ax

        forest(starts[0], (-10, 2), [-10, -5, 0], [
            ("Diligent steering", diligent, "delta", BLUE, "s"),
            ("Random steering", random, "delta", AQUA, "^")],
            "Accuracy change vs no steering (pp)")
        forest(starts[1], (0, 36), [0, 10, 20, 30], [
            ("Harmful requests", harmful_refusal, "delta", INK2, "o"),
            ("Harmless requests", desperate["overrefusal_monitoring.csv"],
             "delta_vs_no_steer", ORANGE, "D")],
            "Refusal change vs a neutral cue (pp)")

        ax = fig.add_axes([starts[2], .22, .280, .45])
        ax.set(xlim=(0, 40), ylim=(-.45, 2.9), yticks=[], xticks=[0, 20, 40])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        for y, label, rate, color in zip((2, 1, 0), ("No cue", "Neutral cue", "Calm cue"),
                                       rates, (GREY, INK2, BLUE)):
            ax.barh(y, rate, height=.28, color=color, zorder=2)
            ax.text(0, y + .27, label, fontsize=10.5, color=INK)
            ax.text(rate + 1.1, y, f"{rate:.1f}%", va="center", fontsize=10.5, color=INK)
        ax.set_xlabel("Agreement with false claims (%)", labelpad=7)

        for suffix in ("png", "svg", "pdf"):
            fig.savefig(OUT / f"mats_overview.{suffix}", dpi=300,
                        bbox_inches=fig.bbox_inches, pad_inches=0)
        plt.close(fig)


# =================================================================================
# Fig 2: the surviving effects, each next to the measurement that undercuts it
# =================================================================================
def fig1_safety_effects() -> None:
    sp = rows("safety_primary.csv")
    om = rows("overrefusal_monitoring.csv")
    djc = {(r["model"], r["state"]): r for r in rows("dose_joint_controls.csv")
           if r["alpha"] == "0.5" and r["signal"] == "unsafe_compliance"}
    dc = {(r["model"], r["state"]): r for r in rows("dose_curves.csv")
          if r["alpha"] == "0.5" and r["signal"] == "unsafe_compliance"}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.65))

    def status(r):
        if r["status"] == "incoherent":
            return "excluded"
        return "holm" if r["retained"] == "True" else "ns"

    ax = axes[0]
    size_axis(ax)
    for r in sp:
        if r["condition"] == "stage4_calm_high__none" and r["model"] in XS:
            y = pp(r["delta_vs_no_steer"])
            point(ax, XS[r["model"]], y, pp(r["ci_low"]), pp(r["ci_high"]), INK2, "o", status(r),
                  fmt(y), side="left" if r["model"].endswith("14B") else "right")
    ax.set_title("2a   Calm cue", loc="left", pad=15)
    ax.set_ylabel("change vs control (percentage points)")
    ax.text(0, 1.02, "agreement with false claims", transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.set_ylim(-15, 8)

    ax = axes[1]
    size_axis(ax)
    for r in sp:
        if r["condition"] == "stage4_desperate_medium__none" and r["model"] in XS:
            y = pp(r["delta_vs_no_steer"])
            point(ax, XS[r["model"]], y, pp(r["ci_low"]), pp(r["ci_high"]), INK2, "o", status(r),
                  fmt(y) if r["model"].endswith("8B") else None, dx=-0.035)
    for r in om:
        if r["condition"] == "stage4_desperate_medium__none" and r["model"] in XS:
            y = pp(r["delta_vs_no_steer"])
            point(ax, XS[r["model"]], y, pp(r["ci_low"]), pp(r["ci_high"]), INK, "D", status(r),
                  fmt(y) if r["model"].endswith("8B") else None, dx=0.035)
    ax.text(math.log10(13), 14, "◆ refuses harmless\nrequests", fontsize=ANN, color=INK)
    ax.text(math.log10(13), -14, "● complies with\nharmful ones", fontsize=ANN, color=INK2)
    ax.set_title("2b   Urgent cue", loc="left", pad=15)
    ax.text(0, 1.02, "request judgments", transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.set_ylim(-18, 34)

    ax = axes[2]
    size_axis(ax)
    offs = {"positive": -0.06, "negative": -0.02, "random": 0.02, "shuffled": 0.06}
    for state in ("positive", "negative", "random", "shuffled"):
        name, col, mk = VEC[state]
        for m in SIZES:
            r, j = dc[(m, state)], djc.get((m, state))
            y = pp(r["delta_vs_no_steer"])
            if state == "negative":
                cell = next(s for s in sp if s["model"] == m and s["condition"] == "neutral_length_matched__negative_a0.5")
                st = status(cell)
            elif state == "positive":
                ok = j["coherent"] == "True" and j["direction_consistent"] == "True" and float(j["joint_p_value"]) < 0.05
                st = "explore" if ok else ("excluded" if j["coherent"] != "True" else "ns")
            else:
                st = "ns" if r["coherent"] == "True" else "excluded"
            sz = 46 if state in ("positive", "negative") else 24
            point(ax, XS[m], y, pp(r["ci_low"]), pp(r["ci_high"]), col, mk, st, dx=offs[state], size=sz)
    ax.set_title("2c   Steering", loc="left", pad=15)
    ax.text(0, 1.02, "unsafe compliance", transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.set_ylim(-11, 16)

    fig.tight_layout(w_pad=1.2)
    vec_handles = [Line2D([], [], marker=m, ls="", color=c, markersize=6, label=n) for n, c, m in VEC.values()]
    fig.legend(handles=STATUS_HANDLES + vec_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.14),
               handletextpad=0.3, columnspacing=1.0, fontsize=8.2)
    fig.savefig(OUT / "mats_fig1_safety_effects.png")
    plt.close(fig)


# =================================================================================
# Fig 3: 8B IFEval - paired change vs no steering, and truncation
# =================================================================================
def fig2_ifeval_8b() -> None:
    pdl = {(r["condition"], r["baseline"]): r for r in rows("stage35_ifeval_paired_deltas.csv")
           + rows("stage35_ifeval_random_vs_nosteer.csv") if r["task"] == "ifeval" and r["signal"] == "strict_prompt"}
    summ = {r["condition"]: r for r in rows("stage35_ifeval_combined_summary.csv") if r["task"] == "ifeval"}
    chk = {r["condition"]: r for r in rows("stage35_ifeval_truncation_check.csv")}
    arms = [("positive_a+0.5000", "positive"), ("negative_a-0.5000", "negative"),
            ("random_a+0.5000", "random"), ("shuffled_a+0.5000", "shuffled")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.45, 1]})
    ys = list(range(len(arms)))[::-1]
    n313 = 313
    base313 = int(chk["no_steer"]["correct_on_313"])
    a1.axvline(0, color=AXIS, lw=1.2)
    for y, (c, st) in zip(ys, arms):
        name, col, mk = VEC[st]
        r = pdl[(c, "no_steer")]
        d, lo, hi = pp(r["delta"]), pp(r["ci_low"]), pp(r["ci_high"])
        a1.plot([lo, hi], [y, y], color=col, lw=1.8)
        a1.scatter([d], [y], marker=mk, s=48, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        a1.annotate(fmt(d), (lo, y), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=ANN, color=INK)
        d313 = 100 * (int(chk[c]["correct_on_313"]) - base313) / n313
        a1.scatter([d313], [y], marker="|", s=110, color=INK2, linewidth=1.6, zorder=4)
    a1.set_yticks(ys, [VEC[s][0] for _, s in arms])
    a1.set_xlim(-11, 3)
    a1.set_xlabel("change in IFEval accuracy vs no steering (points)")
    a1.grid(axis="y", visible=False)
    a1.set_title("3a   Instruction-following accuracy", loc="left")
    a1.legend(handles=[Line2D([], [], color=INK2, lw=1.8, marker="o", label="All prompts"),
                       Line2D([], [], color=INK2, lw=0, marker="|", markersize=10, markeredgewidth=1.6,
                              label="All answers finished")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.2, handlelength=1.4)
    nt = pp(summ["no_steer"]["output_truncated_mean"])
    for y, (c, st) in zip(ys, arms):
        t = pp(summ[c]["output_truncated_mean"])
        a2.barh([y], [t], height=0.4, color=VEC[st][1], zorder=2)
        a2.annotate(f"{t:.1f}%", (t, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=ANN, color=INK)
    a2.axvline(nt, color=GREY, lw=2.2)
    a2.text(nt + 0.4, -0.62, f"No steering {nt:.1f}%",
            fontsize=8.2, color=INK2, va="center")
    a2.set_xlim(0, 18)
    a2.set_ylim(-0.9, len(arms) - 0.4)
    a2.set_xlabel("Answers at the length limit (%)")
    a2.grid(axis="y", visible=False)
    a2.set_title("3b   Answers cut off", loc="left")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "mats_fig2_ifeval_8b.png")
    plt.close(fig)


# =================================================================================
# Fig 5: calm effect on the original vs the calibrated item set
# =================================================================================
def fig3_item_set_flip() -> None:
    s6 = {r["model"]: r for r in rows("stage6_component_comparisons.csv")
          if r["condition"] == "stage4_calm_high__none" and r["signal"] == "sycophancy"
          and r["baseline"] == "neutral_length_matched__none"}
    b6 = {r["model"]: float(r["rate"]) for r in rows("stage6_admission_status.csv") if r["signal"] == "sycophancy"}
    s7 = {r["model"]: r for r in rows("safety_primary.csv")
          if r["condition"] == "stage4_calm_high__none" and r["signal"] == "sycophancy"}
    b7 = {r["model"]: float(r["held_out_rate"]) for r in rows("base_rates.csv")
          if r["signal"] == "sycophancy" and r["held_out_rate"]}
    fig, ax = plt.subplots(figsize=(W, 2.7))
    ax.axvline(0, color=AXIS, lw=1.2)
    for i, m in enumerate(SIZES):
        y = 2 - i
        e6, l6, h6 = pp(s6[m]["delta"]), pp(s6[m]["ci_low"]), pp(s6[m]["ci_high"])
        e7, l7, h7 = pp(s7[m]["delta_vs_no_steer"]), pp(s7[m]["ci_low"]), pp(s7[m]["ci_high"])
        ax.plot([l6, h6], [y + 0.13, y + 0.13], color=GREY, lw=1.6)
        ax.plot([l7, h7], [y - 0.13, y - 0.13], color=BLUE, lw=1.6)
        if abs(e7 - e6) > 1.5:
            ax.annotate("", xy=(e7, y - 0.13), xytext=(e6, y + 0.13),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, shrinkA=5, shrinkB=6, mutation_scale=10))
        ax.scatter([e6], [y + 0.13], s=46, color=GREY, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        sig7 = s7[m]["retained"] == "True"
        ax.scatter([e7], [y - 0.13], s=50, zorder=4, linewidth=1.6,
                   color=BLUE if sig7 else SURFACE, edgecolor=SURFACE if sig7 else BLUE)
        ax.annotate(fmt(e6), (h6, y + 0.13), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=ANN, color=INK2)
        ax.annotate(fmt(e7) + ("" if sig7 else " (n.s.)"), (h7, y - 0.13), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=ANN, color=INK)
        ax.text(-16.6, y, LAB[m], va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(9.5, y, f"{b6[m]:.2f} → {b7[m]:.2f}", va="center", fontsize=ANN, color=INK2)
    ax.set_yticks([])
    ax.set_ylim(-0.55, 2.5)
    ax.text(9.5, 2.42, "Neutral agreement", fontsize=8.2, color=INK2)
    ax.set_xlim(-17, 16)
    ax.set_xlabel("calm-cue effect on agreement with false claims (points)")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="-", color=GREY, markersize=6, label="Original items"),
        Line2D([], [], marker="o", ls="-", color=BLUE, markersize=6, label="Calibrated items"),
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=BLUE, markersize=6,
               label="Inconclusive after correction"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=8.4)
    fig.savefig(OUT / "mats_fig4_item_set_flip.png")
    plt.close(fig)


# =================================================================================
# Fig 6: agreement with false claims by item family and model (dev split) + mix weights
# =================================================================================
def fig4_family_rates() -> None:
    tm = [r for r in rows("tier_mixes.csv") if r["signal"] == "sycophancy"]
    fams = ["semiprime_primality", "modular_exponentiation", "matrix_determinant", "calendar_weekday",
            "weighted_graph_path", "synthetic_policy_exception"]
    fam_lab = ["primality", "modular power", "determinant", "calendar weekday", "graph shortest path",
               "policy with exception"]
    models = SIZES + ["us.anthropic.claude-sonnet-5"]
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE] + SEQ)
    fig, ax = plt.subplots(figsize=(W * 0.74, 2.9))
    for j, m in enumerate(models):
        for i, f in enumerate(fams):
            r = next(x for x in tm if x["model"] == m and x["difficulty_tier_name"] == f)
            v, w = float(r["dev_rate"]), float(r["mixture_weight"])
            used = w > 0 and m != models[-1]
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.04), 0.96, 0.92, color=cmap(v), lw=0))
            light = v > 0.55
            ax.text(j + 0.5, i + 0.42 if used else i + 0.5, f"{100 * v:.0f}%", ha="center", va="center",
                    fontsize=9.2, color="#ffffff" if light else INK, fontweight="bold" if used else "normal")
            if used:
                ax.text(j + 0.5, i + 0.76, f"{100 * w:.0f}% of test mix", ha="center", va="center", fontsize=7.3,
                        color="#ffffff" if light else (INK if v > 0.3 else INK2))
    ax.set_xlim(0, len(models))
    ax.set_ylim(len(fams), 0)
    ax.set_xticks([j + 0.5 for j in range(len(models))], [LAB[m] for m in models])
    ax.set_yticks([i + 0.5 for i in range(len(fams))], fam_lab)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    fig.savefig(OUT / "mats_fig5_family_rates.png")
    plt.close(fig)


# =================================================================================
# Fig 7: probe AUC by layer, with the bigram text baseline
# =================================================================================
def fig5_probe_layers() -> None:
    big = {r["pair"]: float(r["bigram_auc"]) for r in rows("bigram_baseline.csv")}
    specs = [("stage35_qwen3_8b_probe_v1_layer_profile.csv", "Qwen3-8B", 23),
             ("stage35_qwen3_14b_probe_v1_layer_profile.csv", "Qwen3-14B", 27)]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True)
    for ax, panel, (fn, name, steer) in zip(axes, ("7a", "7b"), specs):
        rr = [r for r in rows(fn) if r["policy"] == "last"]
        L = sorted({int(r["layer"]) for r in rr})
        dil = {int(r["layer"]): float(r["direction_auc"]) for r in rr if r["pair"] == "diligence"}
        sing = {l: [float(r["direction_auc"]) for r in rr if int(r["layer"]) == l and r["pair"] != "diligence"] for l in L}
        sctl = {l: max(max(float(r["shuffled_auc_oriented"]), float(r["random_auc_oriented"]))
                       for r in rr if int(r["layer"]) == l and r["pair"] != "diligence") for l in L}
        dctl = {int(r["layer"]): max(float(r["shuffled_auc_oriented"]), float(r["random_auc_oriented"]))
                for r in rr if r["pair"] == "diligence"}
        ax.axhline(min(big.values()), color=INK, lw=1.0)
        ax.axhline(0.5, color=AXIS, lw=1.2)
        ax.fill_between(L, [0.5] * len(L), [sctl[l] for l in L], color=GRID, alpha=0.9, lw=0, hatch="////",
                        edgecolor=SURFACE)
        ax.fill_between(L, [min(sing[l]) for l in L], [max(sing[l]) for l in L], color=INK2, alpha=0.12, lw=0)
        ax.plot(L, [sum(sing[l]) / len(sing[l]) for l in L], color=INK2, marker="o", ms=3.5, lw=1.5)
        ax.plot(L, [dil[l] for l in L], color=BLUE, marker="o", ms=4, markeredgecolor=SURFACE)
        ax.plot(L, [dctl[l] for l in L], color=GREY, lw=1.3)
        ax.axvline(steer, color=MUTED, lw=0.8)
        ax.set_title(f"{panel}   {name}", loc="left")
        ax.set_xticks(L[::2])
        ax.set_xlabel("layer")
        ax.set_ylim(0.45, 1.09)
        ax.grid(axis="x", visible=False)
        ax.annotate(f"steered: {dil[steer]:.2f}", (steer, dil[steer]), xytext=(0, -40), textcoords="offset points",
                    ha="center", fontsize=8.2, color=INK, bbox=dict(fc=SURFACE, ec="none", pad=1.2),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    axes[0].set_ylabel("held-out AUC")
    fig.tight_layout(w_pad=1.0)
    fig.legend(handles=[
        Line2D([], [], color=INK2, marker="o", ms=3.5, lw=1.5, label="5 single pairs: mean"),
        Patch(color=INK2, alpha=0.12, label="5 single pairs: range"),
        Line2D([], [], color=BLUE, marker="o", ms=4, lw=2, label="combined direction (steered)"),
        Line2D([], [], color=GREY, lw=1.3, label="its shuffled / random control"),
        Patch(facecolor=GRID, hatch="////", edgecolor=SURFACE, label="single pairs' controls"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), fontsize=8.2)
    fig.savefig(OUT / "mats_fig6_probe_layers.png")
    plt.close(fig)


# =================================================================================
# Fig 4: every cue x every model on agreement with false claims (exploratory grid)
# =================================================================================
def fig6_cue_grid() -> None:
    cc = [r for r in rows("cue_crosschecks.csv") if r["outcome"] == "agreement_with_false_claims"]
    cues = ["neutral_vs_no_cue", "calm_low", "calm_medium", "calm_high", "desperate_low", "desperate_medium",
            "desperate_high"]
    cue_lab = {"neutral_vs_no_cue": "neutral cue vs no cue"}
    models = ["8b", "14b", "32b"]
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#9ec5f4", MID, "#f2a3a2", RED])
    lim = 20.0
    fig, ax = plt.subplots(figsize=(W * 0.66, 3.3))
    for j, m in enumerate(models):
        for i, c in enumerate(cues):
            r = next(x for x in cc if x["model"] == m and x["cue"] == c)
            v, lo, hi = float(r["delta_pp"]), float(r["ci_low"]), float(r["ci_high"])
            sig = lo > 0 or hi < 0
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.04), 0.96, 0.92, color=cmap((v + lim) / (2 * lim)), lw=0))
            ax.text(j + 0.5, i + 0.5, fmt(v) + ("" if sig else " ns"), ha="center", va="center", fontsize=9.2,
                    color="#ffffff" if abs(v) > 12 else (INK if sig else MUTED), fontweight="bold" if sig else "normal")
            if c == "calm_high":
                ax.add_patch(plt.Rectangle((j + 0.02, i + 0.04), 0.96, 0.92, fill=False, ec=INK, lw=1.4))
    ax.set_xlim(0, 3)
    ax.set_ylim(len(cues), 0)
    ax.set_xticks([j + 0.5 for j in range(3)], ["8B", "14B", "32B"])
    ax.set_yticks([i + 0.5 for i in range(len(cues))], [cue_lab.get(c, c.replace("_", " ")) for c in cues])
    ax.axhline(1, color=INK2, lw=1.0)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    fig.savefig(OUT / "mats_fig3_cue_grid.png")
    plt.close(fig)


# =================================================================================
# Fig A1: 8B dose-response - unsafe compliance, over-refusal, truncation
# =================================================================================
def figA1_dose_8b() -> None:
    dc = [r for r in rows("dose_curves.csv") if r["model"] == "Qwen/Qwen3-8B"
          and r["condition"].startswith("neutral_length_matched__")]
    cw = {r["condition"]: r for r in rows("coherence_wall.csv")
          if r["model"] == "Qwen/Qwen3-8B" and r["task"] == "refusal_drift"}
    al = [0.1, 0.2, 0.3, 0.5]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.9))
    ends = {"unsafe_compliance": {}, "overrefusal": {}}
    for ax, sig, title in ((axes[0], "unsafe_compliance", "A1a   Unsafe compliance"),
                           (axes[1], "overrefusal", "A1b   Over-refusal")):
        ax.axhline(0, color=AXIS, lw=1.2)
        for st, (name, col, mk) in VEC.items():
            pts = sorted((float(r["alpha"]), pp(r["delta_vs_no_steer"]), r["coherent"] == "True") for r in dc
                         if r["state"] == st and r["signal"] == sig and float(r["alpha"]) in al)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.8)
            for a, v, ok in pts:
                ax.scatter([a], [v], marker=mk, s=26, zorder=3, linewidth=1.2,
                           color=col if ok else SURFACE, edgecolor=SURFACE if ok else col)
            ends[sig][st] = pts[-1][1]
        ax.set_title(title, loc="left")
        ax.set_xticks(al)
        ax.set_xlim(0.07, 0.72)
        ax.set_xlabel("α")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("change vs no steering (points)")
    for st in ("negative", "random"):
        axes[0].annotate(f"{VEC[st][0].split()[0]} {fmt(ends['unsafe_compliance'][st])}",
                         (0.5, ends["unsafe_compliance"][st]), xytext=(6, 0), textcoords="offset points",
                         fontsize=7.8, color=INK, va="center")
    for st, dy in (("random", 4), ("shuffled", -5)):
        axes[1].annotate(f"{VEC[st][0].split()[0]} {fmt(ends['overrefusal'][st])}", (0.5, ends["overrefusal"][st]),
                         xytext=(6, dy), textcoords="offset points", fontsize=7.8, color=INK, va="center")
    ax = axes[2]
    for st, (name, col, mk) in VEC.items():
        pts = [(a, pp(cw[f"neutral_length_matched__{st}_a{a:g}"]["output_truncated_mean"])) for a in al]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=4.5, markeredgecolor=SURFACE)
    ax.axhline(2, color=INK2, lw=0.8)
    ax.text(0.1, 2.4, "2% limit", fontsize=8, color=INK2)
    ax.annotate("shuffled 7.9%", (0.5, 7.9), xytext=(6, 0), textcoords="offset points", fontsize=7.8, color=INK, va="center")
    ax.set_ylim(-0.5, 10)
    ax.set_xticks(al)
    ax.set_xlim(0.07, 0.72)
    ax.set_xlabel("α")
    ax.set_title("A1c   Answers cut off (%)", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    fig.legend(handles=[Line2D([], [], color=c, marker=m, lw=2, markersize=5, label=n) for n, c, m in VEC.values()]
               + [Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=5,
                         label="open: condition fails coherence")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=8.2)
    fig.savefig(OUT / "mats_figA1_dose_8b.png")
    plt.close(fig)


# =================================================================================
# Fig A2: base rates before / after calibration
# =================================================================================
def figA2_base_rates() -> None:
    b6 = {(r["model"], r["signal"]): float(r["rate"]) for r in rows("stage6_admission_status.csv")}
    b7 = {}
    for r in rows("base_rates.csv"):
        held = r["held_out_rate"]
        b7[(r["model"], r["signal"])] = (float(held) if held else float(r["predicted_dev_rate"]), r["held_out_status"])
    sigs = [("sycophancy", "A2a   False-claim agreement", (0.10, 0.90)),
            ("unsafe_compliance", "A2b   Unsafe compliance", (0.10, 0.60)),
            ("overrefusal", "A2c   Over-refusal", (0.10, 0.90))]
    models = SIZES + ["us.anthropic.claude-sonnet-5"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.7), sharey=True)
    for ax, (sig, title, band) in zip(axes, sigs):
        ax.axhspan(*band, color=GRID, alpha=0.55, lw=0)
        for i, m in enumerate(models):
            if (m, sig) in b6:
                ax.scatter([i - 0.13], [b6[(m, sig)]], s=34, color=GREY, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            v, status = b7[(m, sig)]
            ok = status == "admitted"
            ax.scatter([i + 0.13], [v], s=44, zorder=4, linewidth=1.5,
                       color=BLUE if ok else SURFACE, edgecolor=SURFACE if ok else BLUE)
            if not ok:
                ax.annotate(f"{v:.2f}", (i + 0.13, v), xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2)
        ax.set_xticks(range(4), [LAB[m] for m in models], fontsize=8.6)
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("no-steer base rate")
    axes[0].set_ylim(-0.04, 1.0)
    fig.tight_layout()
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=GREY, markersize=6, label="original items"),
        Line2D([], [], marker="o", ls="", color=BLUE, markersize=6, label="calibrated items, admitted"),
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=BLUE, markersize=6,
               label="calibrated, not testable"),
        Line2D([], [], marker="s", ls="", color=GRID, markersize=9, label="admission band"),
    ], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(OUT / "mats_figA2_base_rates.png")
    plt.close(fig)


# =================================================================================
# Fig A3: the 30-cell IFEval family (final calibrated run)
# =================================================================================
def figA3_ifeval_family() -> None:
    ifr = rows("ifeval_primary.csv")
    order = SIZES + ["us.anthropic.claude-sonnet-5"]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.axvline(0, color=AXIS, lw=1.2)
    ax.axvspan(-5, 5, color=GRID, alpha=0.4, lw=0)
    for gi, m in enumerate(order):
        cells = [r for r in ifr if r["model"] == m]
        for j, r in enumerate(cells):
            y = gi + (j - (len(cells) - 1) / 2) * 0.09
            d, lo, hi = pp(r["delta_vs_no_steer"]), pp(r["ci_low"]), pp(r["ci_high"])
            ok = r["coherent"] == "True"
            steer = r["candidate_type"] == "steering"
            col = BLUE if ok else MUTED
            ax.plot([lo, hi], [y, y], color=col, lw=1, alpha=0.6)
            if ok:
                ax.scatter([d], [y], marker="o", s=26, color=BLUE, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            else:
                ax.scatter([d], [y], marker="s" if steer else "X", s=30 if steer else 34, zorder=3,
                           color=SURFACE if steer else MUTED, edgecolor=MUTED, linewidth=1.2)
    ax.set_yticks(range(len(order)), [LAB[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(-17, 7)
    ax.set_xlabel("change in IFEval accuracy vs neutral cue / no steering (points), n = 433")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, markersize=6, label="cue condition, passes coherence"),
        Line2D([], [], marker="X", ls="", color=MUTED, markersize=7, label="cue condition, excluded"),
        Line2D([], [], marker="s", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=6,
               label="steering condition, excluded (all six)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3)
    fig.savefig(OUT / "mats_figA3_ifeval_family.png")
    plt.close(fig)


# =================================================================================
# Fig A4: shortcut 'effect' that the shuffled control kills (8B pilot), both cue conditions
# =================================================================================
def figA4_shortcut_controls() -> None:
    am = {r["condition"]: float(r["mean"]) for r in rows("stage5_reanalysis_arm_means.csv") if r["signal"] == "shortcut"}
    n = 232
    arms = [("none", "no steering", GREY, "o"), ("random", "random", AQUA, "^"), ("positive", "diligent (+)", BLUE, "s"),
            ("negative", "careless (−)", ORANGE, "o"), ("shuffled", "shuffled-label", YELLOW, "D")]

    def wilson(p):
        z = 1.96
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return 100 * (c - h), 100 * (c + h)

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for ax, cond, title in ((axes[0], "neutral_length_matched", "A4a   Neutral cue"),
                            (axes[1], "neutral_empty", "A4b   No cue")):
        for i, (st, lab, col, mk) in enumerate(arms[::-1]):
            p = am[f"{cond}__{st}"]
            lo, hi = wilson(p)
            ax.plot([lo, hi], [i, i], color=col, lw=1.8)
            ax.scatter([100 * p], [i], marker=mk, s=44, color=col, edgecolor=SURFACE, linewidth=1.4, zorder=3)
            ax.annotate(f"{100 * p:.1f}%", (hi, i), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=8.2, color=INK)
        ax.set_yticks(range(len(arms)), [a[1] for a in arms[::-1]])
        ax.set_xlim(0, 52)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("shortcut rate (%), 95% Wilson CI")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "mats_figA4_shortcut_controls.png")
    plt.close(fig)


if __name__ == "__main__":
    for old in ("mats_fig1_safety_by_size.png", "mats_fig4_probe_layers.png", "mats_fig3_item_set_flip.png",
                "mats_fig4_family_rates.png", "mats_fig5_probe_layers.png", "mats_fig6_cue_grid.png"):
        (OUT / old).unlink(missing_ok=True)
    for f in (fig0_overview, fig1_safety_effects, fig2_ifeval_8b, fig3_item_set_flip, fig4_family_rates, fig5_probe_layers,
              fig6_cue_grid, figA1_dose_8b, figA2_base_rates, figA3_ifeval_family, figA4_shortcut_controls):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("mats_*.png")))
