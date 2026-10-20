PY ?= python
DATA ?= avocado

.PHONY: help install lint test smoke synthetic synthetic-quick splits features baselines sequence sde all-models \
        ablation-sampling ablation-photos logo gradients summary foundations

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install:            ## editable install with dev and feature extras
	$(PY) -m pip install -e ".[dev,features]"
	pre-commit install

lint:               ## ruff check + format check
	ruff check . && ruff format --check .

test:               ## CPU test suite
	$(PY) -m pytest

smoke:              ## every model, one batch, synthetic (+ avocado if data/avocado exists)
	$(PY) scripts/smoke_all.py

synthetic:          ## fake-fruit recovery at full scale -> results/synthetic/
	$(PY) scripts/synthetic_recovery.py

synthetic-quick:    ## same, ~1 min
	$(PY) scripts/synthetic_recovery.py --quick

splits:             ## write the item-level split files for $(DATA)
	$(PY) scripts/make_splits.py data=$(DATA)
	$(PY) scripts/make_splits.py data=$(DATA) splits.mode=logo splits.held_out=T10
	$(PY) scripts/make_splits.py data=$(DATA) splits.mode=logo splits.held_out=T20
	$(PY) scripts/make_splits.py data=$(DATA) splits.mode=logo splits.held_out=Tamb

features:           ## cache ResNet-18 features for the avocado photos (needs [features] extra)
	$(PY) scripts/extract_features.py --data avocado --root data/avocado --out data/features/avocado_resnet18.npz

baselines:          ## per-photo baselines on $(DATA)
	for m in stage_linear direct_reg discrete_surv; do $(PY) scripts/train.py model=$$m data=$(DATA); done

sequence:           ## GRU / GRU-D / ODE-RNN on $(DATA)
	for m in gru_gap gru_d ode_rnn; do $(PY) scripts/train.py model=$$m data=$(DATA); done

sde:                ## latent ODE and latent SDE on $(DATA)
	for m in latent_ode latent_sde; do $(PY) scripts/train.py model=$$m data=$(DATA); done

all-models: baselines sequence sde   ## everything above

ablation-sampling:  ## direction 3: photos every 2 / 3 days
	for k in 2 3; do for m in gru_d latent_ode latent_sde; do $(PY) scripts/train.py model=$$m data=$(DATA) +experiment=sampling_every$$k; done; done

ablation-photos:    ## direction 4: metrics vs number of visible photos, from finished runs
	for m in gru_d latent_ode latent_sde; do $(PY) scripts/evaluate.py results/runs/$${m}_$(DATA) --n-obs 1 2 3 5 8; done

logo:               ## train on two storage groups, test on the third
	for g in T10 T20 Tamb; do for m in direct_reg gru_d latent_sde; do $(PY) scripts/train.py model=$$m data=$(DATA) +experiment=logo splits.held_out=$$g; done; done

gradients:          ## backprop vs checkpointing vs adjoint for the latent SDE
	for m in backprop checkpoint adjoint; do $(PY) scripts/train.py model=latent_sde data=$(DATA) +experiment=gradient_methods model.method=$$m; done

summary:            ## results/summary.md from every finished run
	$(PY) scripts/summarize.py

foundations:        ## the original nsde experiments (gradient benchmark, local-vol calibration)
	$(PY) experiments/adjoint_tradeoffs.py --quick
	$(PY) experiments/local_vol_calibration.py
