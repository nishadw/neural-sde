# Lecture: Neural SDEs: Learning Random Dynamics

A graduate lecture for UMD CMSC 838B (Differentiable Programming), Fall 2026, with produce decay as the running
example. It covers neural ODEs as continuous ResNets, backprop vs the adjoint method, Brownian motion and Itô's
lemma, Euler–Maruyama, and pathwise SDE training with fixed noise. The appendix covers latent SDEs, the adjoint
derivation, an Arrhenius fit, and Brownian-bridge replay.

| File | Contents |
|---|---|
| `neural_sde_lecture.pptx` | 16 presented slides + appendix, with speaker notes |
| `neural_sde_demo.py` | Live demo: one gradient via the stochastic adjoint and via backprop ([recording](../assets/demo.mp4)) |
| `build/` | Figure and deck generators from earlier versions (`python figs.py`, `node build_deck.js`) |

**Figure credits.** The paper figures are from Xavier, Rodrigues & Silva, *Foods* 13(8):1150, 2024,
[doi:10.3390/foods13081150](https://doi.org/10.3390/foods13081150), CC BY 3.0. Some are cropped.
