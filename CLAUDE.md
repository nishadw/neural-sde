# ripen

Distributional shelf-life prediction from produce photos (latent neural SDEs, JAX/equinox/diffrax, Hydra configs).
Start with README.md, then docs/experiment_plan.md.

## Cluster
On the UMIACS Nexus class cluster (`nexusclass00/01`), load the `umiacs` skill (`.claude/skills/umiacs/SKILL.md`)
before running anything. Nothing heavier than editing, git or reading logs runs on the login node; every
compute step goes through `run_umiacs.sh` (sbatch) or `srun`.

## Conventions
- Tests: `pytest -m "not slow"` (CPU, `JAX_PLATFORMS=cpu`); lint: `ruff check . && ruff format --check .` (line length 120).
- Raw data never enters git (`data/*` is ignored except `data/README.md` and `data/manifests/`); split files and summaries do.
- Avocado data: `data/avocado/Avocado Ripening Dataset.xlsx` + `data/avocado/images/*.jpg` (14,710 photos, 478 fruit);
  `scripts/download_avocado.py` resumes. The spreadsheet has 14,722 rows; 12 photos are missing from the release.
