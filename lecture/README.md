# Lecture: Backprop Through Continuous Time

Student lecture for CMSC 838B (Differentiable Programming), University of Maryland, Fall 2026.

| File | What it is |
|---|---|
| `neural_sde_lecture.pdf` | Slides (19 + 7 backup); running example: avocado shelf life |
| `neural_sde_lecture.pptx` | Same slides with timestamped speaker notes and "if asked" answers (~24 min talk + 5 min discussion) |
| `neural_sde_reading_guide.docx` | Class reading list, prep plan, every slide equation in plain words, likely questions, and the placeholder to-do list |
| `neural_sde_demo.py` | Live demo: the same gradient via the stochastic adjoint and via backprop |
| [`../assets/demo.mp4`](../assets/demo.mp4) | Recorded demo (52 s): this script's output, a live calibration, and the benchmark results. Use it if the live demo can't run |
| `build/` | Scripts that generate the figures, deck and guide (`python figs.py`, then `node build_deck.js` and `node build_doc.js`; needs `pptxgenjs` and `docx` from npm) |
