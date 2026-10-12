"""Render assets/demo.mp4 from real outputs: demo script, pytest, calibration snapshots, adjoint sweep. Run via make.sh."""
import json, pathlib, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import NullLocator

HERE = pathlib.Path(__file__).parent / "_cache"
REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(sys.argv[1])
FPS = 30
W, H = 12.8, 7.2

INK, MUTED, FAINT, RED, GREY = "#1F2328", "#6B7280", "#9AA3AF", "#C8102E", "#6B7280"
TERM_BG, TERM_FG, TERM_DIM, TERM_OK = "#1F2228", "#E5E7EB", "#9AA3AF", "#7EE787"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#D1D5DB",
    "axes.labelcolor": "#374151", "xtick.color": "#4B5563", "ytick.color": "#4B5563",
    "axes.titleweight": "bold", "axes.titlesize": 14, "legend.frameon": False,
})

calib = json.load(open(HERE / "calib_frames.json"))
final = json.load(open(REPO / "results" / "local_vol_calibration.json"))
sweep = json.load(open(REPO / "results" / "adjoint_tradeoffs.json"))
demo_out = (HERE / "demo_out.txt").read_text().splitlines()
demo_out = [l for l in demo_out if "cpu" not in l]          # drop the `time` line
pytest_out = [l for l in (HERE / "pytest_out.txt").read_text().splitlines() if l.strip()]

ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                       "-s", f"{int(W*100)}x{int(H*100)}", "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(OUT)],
                      stdin=subprocess.PIPE)


def emit(fig, n=1):
    fig.canvas.draw()
    buf = bytes(fig.canvas.buffer_rgba())
    for _ in range(n):
        ff.stdin.write(buf)


def new_fig():
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor="white")
    return fig


def caption(fig, title, sub=None):
    fig.text(0.05, 0.93, title, fontsize=22, weight="bold", color=INK, va="top")
    if sub:
        fig.text(0.05, 0.865, sub, fontsize=14, color=MUTED, va="top")


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- 1. title
def title_card(seconds=4.0):
    for f in range(int(seconds * FPS)):
        a = ease(f / (0.8 * FPS))
        fig = new_fig()
        fig.text(0.07, 0.60, "neural-sde", fontsize=54, weight="bold", color=INK, alpha=a)
        fig.text(0.07, 0.50, "Backpropagating through stochastic differential equation solvers",
                 fontsize=20, color=MUTED, alpha=a)
        fig.text(0.07, 0.40, "JAX · Diffrax · Equinox     companion to a CMSC 838B lecture (UMD, Fall 2026)",
                 fontsize=14, color=FAINT, alpha=a)
        fig.add_artist(plt.Line2D([0.07, 0.07 + 0.25 * ease(f / FPS)], [0.36, 0.36], color=RED, lw=3))
        fig.text(0.07, 0.12, "Every number in this video comes from a real run of the repo's code.",
                 fontsize=13, color=FAINT, alpha=ease((f - FPS) / FPS), style="italic")
        emit(fig); plt.close(fig)


# ---------------------------------------------------------------- 2. terminal
def terminal(title, sub, cmd, output, highlight, hold=3.5):
    """Type `cmd`, reveal `output` line by line, then hold with `highlight` lines marked."""
    type_frames = int(len(cmd) / 30 * FPS) + 1         # 30 chars/s
    line_frames = 3
    total = int(0.5 * FPS) + type_frames + int(0.6 * FPS) + line_frames * len(output) + int(hold * FPS)
    for f in range(total):
        fig = new_fig()
        caption(fig, title, sub)
        ax = fig.add_axes([0.05, 0.08, 0.90, 0.72])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=TERM_BG, transform=ax.transAxes, zorder=0))
        for i, c in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
            ax.add_patch(plt.Circle((0.02 + i * 0.022, 0.955), 0.008, color=c, transform=ax.transAxes))
        t = f - int(0.5 * FPS)
        n_chars = max(0, min(len(cmd), int(t / FPS * 30)))
        cursor = "▌" if (f // 15) % 2 == 0 and n_chars < len(cmd) else ""
        y = 0.87
        ax.text(0.02, y, "$ ", color=TERM_OK, family="DejaVu Sans Mono", fontsize=14, va="top")
        ax.text(0.04, y, cmd[:n_chars] + cursor, color=TERM_FG, family="DejaVu Sans Mono", fontsize=14, va="top")
        t_out = t - type_frames - int(0.6 * FPS)
        n_lines = max(0, min(len(output), t_out // line_frames + 1)) if t_out >= 0 else 0
        done = n_lines == len(output) and t_out > line_frames * len(output) + 10
        for i, line in enumerate(output[:n_lines]):
            yy = y - 0.07 * (i + 1)
            hit = any(h in line for h in highlight)
            color = TERM_OK if hit and done else TERM_FG if hit else TERM_DIM
            ax.text(0.04, yy, line, color=color, family="DejaVu Sans Mono", fontsize=11.5, va="top",
                    weight="bold" if hit and done else "normal")
        emit(fig); plt.close(fig)


# ---------------------------------------------------------------- 3. calibration
def calibration(seconds=16.0, hold=4.0):
    K = np.array(calib["strikes"]); mask = np.array(calib["train_mask"])
    T = calib["maturities"]; target = 100 * np.array(calib["target_iv"])
    frames = calib["frames"]; hist = np.array(calib["history"])
    flat = float(np.mean(target[:, mask]))
    n = int(seconds * FPS)
    surf_t, surf_s = np.array(calib["surface_t"]), np.array(calib["surface_s"])
    ylo, yhi = 12, 30
    for f in range(n + int(hold * FPS)):
        u = min(f / n, 1.0)
        k = min(int(u * (len(frames) - 1) + 0.5), len(frames) - 1)
        fr = frames[k]; it = fr["iter"]
        iv = 100 * np.array(fr["iv"])
        fig = new_fig()
        caption(fig, "Calibrating a neural local-volatility SDE to a Heston smile",
                "Gradients flow through the SDE solver and the Monte Carlo average. "
                "Open circles are strikes the model never trained on.")
        for j in range(3):
            ax = fig.add_axes([0.05 + j * 0.235, 0.47, 0.20, 0.30])
            ax.axhline(flat, color=FAINT, ls="--", lw=1.2, label="Flat Black-Scholes")
            ok = np.isfinite(iv[j])
            ax.plot(K[ok], iv[j][ok], color=RED, lw=2.4, label="Neural local-vol SDE")
            ax.plot(K[mask], target[j, mask], "o", color=INK, ms=5, label="Heston quotes (train)")
            ax.plot(K[~mask], target[j, ~mask], "o", mfc="white", mec=INK, ms=5, label="Heston quotes (held out)")
            ax.set(title=f"T = {T[j]:g} yr", ylim=(ylo, yhi), xlabel="strike / spot")
            ax.title.set_fontsize(13)
            if j == 0:
                ax.set_ylabel("implied vol (%)")
        ax = fig.add_axes([0.76, 0.47, 0.20, 0.30])
        im = ax.pcolormesh(surf_s, surf_t, 100 * np.array(fr["surface"]), cmap="Reds", shading="auto", vmin=5, vmax=45)
        ax.set(title="Learned σθ(t, S)", xlabel="S / S0", ylabel="t (yr)")
        ax.title.set_fontsize(13)
        # loss curve
        ax = fig.add_axes([0.09, 0.08, 0.38, 0.26])
        ax.plot(np.arange(len(hist))[:max(it, 1)], hist[:max(it, 1)], color=RED, lw=1)
        ax.set(yscale="log", xlim=(0, len(hist)), ylim=(hist.min() * 0.8, hist.max() * 1.2),
               xlabel="Adam iteration", ylabel="≈ IV RMSE (vol pts)")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # handles for legend
        h, l = fig.axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.52, 0.10), ncol=2, fontsize=11.5)
        rm = np.sqrt(np.nanmean((iv - target) ** 2))
        fig.text(0.53, 0.30, f"iteration {it:3d} / 500", fontsize=18, weight="bold", color=INK, family="DejaVu Sans Mono")
        fig.text(0.53, 0.25, f"smile RMSE  {rm:5.2f} vol pts", fontsize=15, color=RED, family="DejaVu Sans Mono")
        if f >= n:
            a = ease((f - n) / (0.6 * FPS))
            r = final["iv_rmse_vol_pts"]
            fig.text(0.79, 0.30, f"Final (524k fresh paths)", fontsize=12, color=MUTED, alpha=a)
            fig.text(0.79, 0.255, f"train      {r['neural_lv_train']:.2f}", fontsize=13, color=INK, alpha=a, family="DejaVu Sans Mono")
            fig.text(0.79, 0.22, f"held out   {r['neural_lv_held_out']:.2f}", fontsize=13, color=INK, alpha=a, family="DejaVu Sans Mono")
            fig.text(0.79, 0.185, f"flat BS    {r['flat_black_scholes_train']:.2f}", fontsize=13, color=FAINT, alpha=a, family="DejaVu Sans Mono")
        emit(fig); plt.close(fig)


# ---------------------------------------------------------------- 4. adjoint trade-offs
def tradeoffs(seconds=9.0, hold=5.0):
    steps = np.array(sweep["backprop"]["steps"])
    methods = [("backprop", "Backprop", INK, "o"), ("checkpoint", "Recursive checkpointing", GREY, "s"),
               ("adjoint", "Stochastic adjoint", RED, "^")]
    panels = [("temp_mb", "Memory per gradient (MB)"), ("seconds", "Time per gradient (s)"),
              ("rel_err", "Adjoint error vs. backprop")]
    n = int(seconds * FPS)
    notes = ["Adjoint: flat, but a 154 MB constant",
             "Adjoint: ~64× slower than backprop",
             "Adjoint: approximate, error ~ 1/steps"]
    for f in range(n + int(hold * FPS)):
        fig = new_fig()
        caption(fig, "How should you differentiate an SDE solve?",
                "Neural SDE, 16-dim state, 32 paths, reversible Heun. Three ways to get the same gradient.")
        for p, (key, ttl) in enumerate(panels):
            ax = fig.add_axes([0.06 + p * 0.32, 0.20, 0.26, 0.55])
            u = ease((f - p * n / 3) / (n / 3))
            for m, lab, c, mk in methods:
                ys = np.array(sweep[m][key], dtype=float)
                if key == "rel_err" and m != "adjoint":
                    continue
                k = u * (len(steps) - 1)
                i = int(k)
                xs_ = list(steps[:i + 1]); ys_ = list(ys[:i + 1])
                if i < len(steps) - 1 and u > 0:
                    fr = k - i
                    xs_.append(np.exp(np.log(steps[i]) * (1 - fr) + np.log(steps[i + 1]) * fr))
                    ys_.append(np.exp(np.log(ys[i]) * (1 - fr) + np.log(ys[i + 1]) * fr))
                if u > 0:
                    ax.plot(xs_, ys_, color=c, lw=2.2, marker=mk, ms=6, markevery=list(range(i + 1)), label=lab)
            ax.set(xscale="log", yscale="log", xlabel="solver steps", title=ttl)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(steps); ax.set_xticklabels([str(s) for s in steps])
            ax.set_xlim(steps[0] * 0.8, steps[-1] * 1.25)
            allv = np.concatenate([np.array(sweep[m][key], float) for m, *_ in methods
                                   if not (key == "rel_err" and m != "adjoint")])
            ax.set_ylim(allv.min() * 0.6, allv.max() * 1.8)
            a = ease((f - (p + 1) * n / 3) / (0.5 * FPS))
            fig.text(0.06 + p * 0.32, 0.085, notes[p], fontsize=12, color=RED if p < 3 else INK, alpha=a)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=c, marker=mk, lw=2.2, label=lab) for _, lab, c, mk in methods]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.835), ncol=3, fontsize=12)
        if f >= n:
            a = ease((f - n - FPS) / (0.6 * FPS))
            fig.text(0.5, 0.025, "Verdict: recursive checkpointing — exact gradients, ~1 MB at 512 steps, within 1.5× of backprop's speed.",
                     ha="center", fontsize=13, weight="bold", color=INK, alpha=a)
        emit(fig); plt.close(fig)


# ---------------------------------------------------------------- 5. end card
def end_card(seconds=4.5):
    for f in range(int(seconds * FPS)):
        a = ease(f / (0.6 * FPS))
        fig = new_fig()
        fig.text(0.07, 0.62, "neural-sde", fontsize=40, weight="bold", color=INK, alpha=a)
        rows = [("Neural local-vol fit, held-out strikes", "0.26 vol pts  (flat BS: 3.87)"),
                ("Gradient memory @ 512 steps", "backprop 10.3 MB · checkpoint 1.0 MB · adjoint 154 MB"),
                ("Adjoint vs. backprop gradient", "7.4e-5 → 1.8e-6 rel. error, 16 → 512 steps")]
        for i, (k, v) in enumerate(rows):
            fig.text(0.07, 0.50 - i * 0.07, k, fontsize=15, color=MUTED, alpha=a)
            fig.text(0.47, 0.50 - i * 0.07, v, fontsize=15, color=INK, alpha=a)
        fig.text(0.07, 0.18, "github.com/nishadw/neural-sde", fontsize=18, color=RED, weight="bold", alpha=a)
        emit(fig); plt.close(fig)


title_card()
terminal("Live demo: two ways to differentiate one SDE solve",
         "lecture/neural_sde_demo.py — stochastic adjoint vs. backprop through the solver",
         "python lecture/neural_sde_demo.py", demo_out, ["max abs diff"])
terminal("The test suite", "Gradient methods agree; Black-Scholes, implied vol and Heston check out",
         "pytest", pytest_out, ["passed"], hold=2.0)
calibration()
tradeoffs()
end_card()
ff.stdin.close(); ff.wait()
print("wrote", OUT)
