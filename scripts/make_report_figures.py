"""Figures for report/REPORT.md, read from results/runs/*/metrics_test.json (run after scripts/summarize.py).

python scripts/make_report_figures.py            # writes report/figures/*.png
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "report" / "figures"
BEFORE = ROOT / "report" / "before_fix"

# reference categorical slots 1-3 (validated all-pairs), muted gray for context bars, text inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#b4b2ab", "#0b0b0b", "#52514e", "#e6e5e0"
NAMES = {
    "stage_linear": "stage→linear (published)",
    "direct_reg": "direct regressor",
    "discrete_surv": "discrete survival",
    "gru_gap": "GRU + gap",
    "gru_d": "GRU-D",
    "ode_rnn": "ODE-RNN",
    "latent_ode": "latent ODE",
    "latent_sde": "latent SDE (ours)",
    "fpt_filter": "FPT filter (ours)",
}
MODELS = list(NAMES)


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def metrics(run: str, split: str = "overall", path: Path | None = None) -> dict | None:
    p = path or RUNS / run / "metrics_test.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(split)


def label_bars(ax, bars, fmt="{:.2f}", horizontal=True):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK2)


def fig_e1():
    rows = [(m, metrics(f"{m}_avocado")) for m in MODELS]
    rows = [(m, r) for m, r in rows if r]
    names = [NAMES[m] for m, _ in rows][::-1]
    colors = [BLUE if m in ("latent_sde", "fpt_filter") else GRAY for m, _ in rows][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for ax, key, title, fmt in [
        (axes[0], "crps", "CRPS (days) ↓", "{:.2f}"),
        (axes[1], "coverage80", "80% coverage (dashed: target)", "{:.2f}"),
        (axes[2], "cindex", "C-index ↑", "{:.3f}"),
    ]:
        vals = [r[key] for _, r in rows][::-1]
        bars = ax.barh(names, vals, color=colors, height=0.6)
        label_bars(ax, bars, fmt)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        if key == "coverage80":
            ax.axvline(0.8, color=INK2, lw=1, ls="--")
            ax.set_xlim(0, 1.1)
        if key == "cindex":
            ax.set_xlim(0.5, 1.0)
    fig.suptitle("E1 · Mendeley avocado test set (905 prediction instances)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "e1_avocado.png")
    plt.close(fig)


def fig_transfer():
    groups = ["T10", "T20", "Tamb"]
    series = [
        ("Arrhenius clock (temperature)", "latent_sde_avocado_logo_{g}", BLUE),
        ("per-group clock", "latent_sde_avocado_logo_{g}_clock_group", ORANGE),
        ("fixed clock", "latent_sde_avocado_logo_{g}_clock_fixed", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = 0.26
    for i, (lab, pat, c) in enumerate(series):
        vals = [(metrics(pat.format(g=g)) or {}).get("mae", float("nan")) for g in groups]
        bars = ax.bar([j + (i - 1) * (w + 0.02) for j in range(3)], vals, w, color=c, label=lab)
        label_bars(ax, bars, "{:.1f}", horizontal=False)
    ax.set_xticks(range(3), [f"held out: {g}" for g in groups])
    ax.set_ylabel("MAE on held-out group (days)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("E3 · Transfer to an unseen storage temperature (latent SDE)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "e3_transfer.png")
    plt.close(fig)


def fig_sampling():
    series = [("GRU-D", "gru_d", GRAY), ("latent ODE", "latent_ode", ORANGE), ("latent SDE (ours)", "latent_sde", BLUE)]
    xs = ["daily", "every 2nd day", "every 3rd day"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in [(axes[0], "crps", "CRPS (days) ↓"), (axes[1], "coverage80", "80% coverage")]:
        for lab, m, c in series:
            runs = [f"{m}_avocado", f"{m}_avocado_every2", f"{m}_avocado_every3"]
            vals = [(metrics(r) or {}).get(key, float("nan")) for r in runs]
            ax.plot(xs, vals, color=c, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=lab)
            ax.annotate(
                lab, (2, vals[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2
            )
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.2, 2.9)
        if key == "coverage80":
            ax.axhline(0.8, color=INK2, lw=1, ls="--")
    axes[0].legend(loc="upper left")
    fig.suptitle("E4 · Training on sparser photo schedules", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "e4_sampling.png")
    plt.close(fig)


def fig_photo_count():
    series = [("GRU-D", "gru_d", GRAY), ("latent ODE", "latent_ode", ORANGE), ("latent SDE (ours)", "latent_sde", BLUE)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for lab, m, c in series:
        p = RUNS / f"{m}_avocado" / "metrics_by_n_obs.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ks = sorted(int(k) for k in d if k.isdigit() and d[k]["n"] >= 20)[:12]
        ax.plot(ks, [d[str(k)]["crps"] for k in ks], color=c, lw=2, marker="o", ms=6, mec="white", mew=1.2, label=lab)
    ax.set_xlabel("photos seen so far")
    ax.set_ylabel("CRPS (days) ↓")
    ax.legend()
    ax.set_title("E4 · Forecast error vs. number of photos observed", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "e4_photo_count.png")
    plt.close(fig)


def fig_pbc2_fix():
    before = {
        "latent ODE": metrics("", path=BEFORE / "latent_ode_pbc2.json"),
        "GRU-D": {"cindex": 0.5, "mae": float("nan")},  # NaN predictions before the fix (C reported as 0.5)
    }
    after = {"latent ODE": metrics("latent_ode_pbc2"), "GRU-D": metrics("gru_d_pbc2")}
    ref = {n: metrics(f"{m}_pbc2") for n, m in [("ODE-RNN", "ode_rnn"), ("discrete survival", "discrete_surv")]}
    names = ["latent ODE", "GRU-D"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.36
    b1 = ax.bar(
        [i - w / 2 - 0.01 for i in range(2)], [before[n]["cindex"] for n in names], w, color=GRAY, label="before fix"
    )
    b2 = ax.bar(
        [i + w / 2 + 0.01 for i in range(2)], [after[n]["cindex"] for n in names], w, color=BLUE, label="after fix"
    )
    label_bars(ax, list(b1) + list(b2), "{:.2f}", horizontal=False)
    # reference lines stop short of their labels; the two reference labels sit above / below their lines
    for (n, r), dy in zip(ref.items(), (-10, 6), strict=True):
        if r:
            ax.hlines(r["cindex"], -0.6, 1.5, color=INK2, lw=1, ls=":")
            ax.annotate(
                f"{n} {r['cindex']:.2f}", (1.52, r["cindex"]), xytext=(0, dy), textcoords="offset points",
                va="center", fontsize=8, color=INK2,
            )  # fmt: skip
    ax.hlines(0.5, -0.6, 1.5, color=INK2, lw=1, ls="--")
    ax.text(1.52, 0.5, "chance", va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(2), names)
    ax.set_xlim(-0.6, 2.2)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("C-index on PBC2 test ↑")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("E5 · PBC2 (liver disease) before and after the two bug fixes", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "e5_pbc2_fix.png")
    plt.close(fig)


def copy_existing():
    for src, dst in [
        (RUNS / "latent_sde_avocado" / "reliability_test.png", "latent_sde_reliability.png"),
        (RUNS / "latent_sde_avocado" / "pred_vs_true_test.png", "latent_sde_pred_vs_true.png"),
        (ROOT / "results" / "synthetic" / "recovery.png", "synthetic_recovery.png"),
    ]:
        if src.exists():
            shutil.copy(src, OUT / dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    fig_e1()
    fig_transfer()
    fig_sampling()
    fig_photo_count()
    fig_pbc2_fix()
    copy_existing()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
