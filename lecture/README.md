# Lecture: Backprop Through Continuous Time

Student lecture for CMSC 838B (Differentiable Programming), University of Maryland, Fall 2026.

| File | What it is |
|---|---|
| `neural_sde_lecture.pdf` | Slides: 22 presented + appendix A1–A10 (not presented). Running example: 'Hass' avocado shelf life, from Xavier et al. (2024) |
| `neural_sde_lecture.pptx` | Same slides with timestamped speaker notes and "if asked" answers, timed to the 25-minute slot (20 min talk + 5 min discussion) |
| `neural_sde_reading_guide.docx` | Class reading list, prep plan, every slide equation in plain words, likely questions, and a one-week fast prep path |
| `neural_sde_practice_script.pdf` | Each presented slide beside its speaker notes, for rehearsing with a timer |
| `neural_sde_demo.py` | Live demo: the same gradient via the stochastic adjoint and via backprop |
| [`../assets/demo.mp4`](../assets/demo.mp4) | Recorded demo (52 s): this script's output, a live calibration, and the benchmark results. Use it if the live demo can't run |
| `build/` | Scripts that generate the figures, deck and guide (`python figs.py`, then `node build_deck.js`, `node build_doc.js` and `python practice_script.py`; needs `pptxgenjs` and `docx` from npm) |

**Figure credits.** `build/img/p_*.jpg` and the paper figures in the deck are from Xavier, Rodrigues & Silva, "Shelf-life management and ripening assessment of 'Hass' avocado using deep learning," *Foods* 13(8):1150, 2024, [doi:10.3390/foods13081150](https://doi.org/10.3390/foods13081150), licensed [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/); some are cropped. Rates, activation energy and first-passage ratios on slides 4, 5, 12 and 13 are derived from the paper's Tables 4 and 6 (worked in appendix A4–A5).
