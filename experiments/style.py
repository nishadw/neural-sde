"""Shared matplotlib style for the result figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "backprop": "#1F2328",
    "checkpoint": "#6B7280",
    "adjoint": "#C8102E",
    "muted": "#9AA3AF",
    "target": "#1F2328",
    "model": "#C8102E",
    "baseline": "#9AA3AF",
}


def setup():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#D1D5DB",
        "axes.labelcolor": "#374151",
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "legend.frameon": False,
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")
