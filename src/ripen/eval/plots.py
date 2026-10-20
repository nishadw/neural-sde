"""Figures: reliability diagram, predicted vs. true days-left, and the synthetic-recovery panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import quantile_from_cdf, reliability_curve  # noqa: E402

INK, ACCENT, MUTED = "#1F2328", "#C8102E", "#9AA3AF"


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def reliability_diagram(F, lo, hi, path: str | Path, title: str = "Reliability"):
    _style()
    bp, bo, bc, ece = reliability_curve(F, lo, hi)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ok = ~np.isnan(bp)
    ax.plot(bp[ok], bo[ok], marker="o", color=ACCENT, lw=1.5)
    for x, y, c in zip(bp[ok], bo[ok], bc[ok], strict=True):
        ax.annotate(str(int(c)), (x, y), fontsize=7, color=INK, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("predicted P(spoiled by h)")
    ax.set_ylabel("observed fraction")
    ax.set_title(f"{title}  (ECE {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def predicted_vs_true(F, hi, path: str | Path, title: str = "Days left"):
    _style()
    unc = ~np.isinf(hi)
    med, q10, q90 = quantile_from_cdf(F, 0.5), quantile_from_cdf(F, 0.1), quantile_from_cdf(F, 0.9)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.errorbar(
        hi[unc],
        med[unc],
        yerr=[med[unc] - q10[unc], q90[unc] - med[unc]],
        fmt="o",
        ms=3,
        color=ACCENT,
        ecolor=MUTED,
        lw=0.6,
    )
    m = max(1.0, float(np.nanmax(hi[unc])) if unc.any() else 1.0)
    ax.plot([0, m], [0, m], color=INK, lw=1)
    ax.set_xlabel("true days left")
    ax.set_ylabel("predicted median (10-90 %)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def recovery_panel(z, drift_true, drift_fit, sigma_true, sigma_fit, t, cdf_true, cdf_fit, path: str | Path):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    axes[0].plot(z, drift_true, color=INK, label="true")
    axes[0].plot(z, drift_fit, color=ACCENT, label="latent SDE")
    axes[0].set_title("drift mu(z)")
    axes[0].set_xlabel("latent ripeness z")
    axes[0].legend()
    axes[1].plot(z, sigma_true, color=INK)
    axes[1].plot(z, sigma_fit, color=ACCENT)
    axes[1].set_title("diffusion sigma(z)")
    axes[1].set_xlabel("latent ripeness z")
    axes[2].plot(t, cdf_true, color=INK)
    axes[2].plot(t, cdf_fit, color=ACCENT)
    axes[2].set_title("P(spoiled by t)")
    axes[2].set_xlabel("days")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
