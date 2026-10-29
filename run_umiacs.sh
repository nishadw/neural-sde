#!/usr/bin/env bash
# One file to run the whole project on UMIACS (or any SLURM cluster, or a workstation with --local).
#
#   bash run_umiacs.sh pipeline         # setup -> data -> all as one chain of sbatch jobs (the usual entry point)
#   bash run_umiacs.sh setup            # sbatch job: venv + editable install + CPU tests (run once)
#   bash run_umiacs.sh data             # sbatch job: download avocado if incomplete, check / prepare every dataset
#   bash run_umiacs.sh all              # submit every experiment with dependencies, then the summary job
#   bash run_umiacs.sh e1|e2|e3|e4|e5|e6|synthetic|summary      # one stage
#   bash run_umiacs.sh all --local      # run everything sequentially in this shell (no sbatch)
#
# setup and data run inside sbatch jobs too (no compute on submission nodes); `--local` runs any stage inline.
# SLURM identifiers: on UMIACS class accounts (nexusclass*) they default to account=class partition=class
# qos=default; elsewhere set RIPEN_ACCOUNT / RIPEN_PARTITION / RIPEN_QOS
# (`sacctmgr show assoc user=$USER format=account,partition,qos` lists them). Everything else is a default you
# can override the same way: RIPEN_ENV (venv path), RIPEN_ROOT (repo), RIPEN_GPU (gres string, default gpu:1),
# RIPEN_PYTHON (interpreter for the venv, >=3.10), RIPEN_AFTER (job ids every submitted job waits for).
# Per-machine values can live in .ripen.env (gitignored), sourced first.
#
# Experiments (see docs/experiment_plan.md):
#   synthetic  fake-fruit recovery at full scale                     results/synthetic/
#   e1         every model on the Mendeley avocado set               results/runs/<model>_avocado
#   e2         exact filter vs PDE-ELBO vs Monte Carlo likelihoods    results/runs/{fpt_filter,latent_sde}_*
#   e3         leave-one-storage-group-out transfer + clock ablation results/runs/*_logo_*
#   e4         sampling-frequency and photo-count ablations          results/runs/*_every{2,3}, metrics_by_n_obs.json
#   e5         second domains: PBC2 and C-MAPSS                      results/runs/<model>_{pbc2,cmapss}
#   e6         A2SL multimodal avocados (photos + microwave)         results/runs/<model>_a2sl
#   summary    results/summary.md
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
[[ -f .ripen.env ]] && source .ripen.env
if [[ "$(hostname)" == nexusclass* ]]; then
  export RIPEN_ACCOUNT="${RIPEN_ACCOUNT:-class}" RIPEN_PARTITION="${RIPEN_PARTITION:-class}" RIPEN_QOS="${RIPEN_QOS:-default}"
fi
export RIPEN_ROOT="${RIPEN_ROOT:-$PWD}"
export RIPEN_ENV="${RIPEN_ENV:-$HOME/envs/ripen}"
export RIPEN_GPU="${RIPEN_GPU:-gpu:1}"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
STAGE="${1:-all}"; LOCAL=0; [[ "${2:-}" == "--local" ]] && LOCAL=1
PY="${RIPEN_ENV}/bin/python"
read -r -a AFTER <<< "${RIPEN_AFTER:-}"
declare -A JOBS
MODELS=(stage_linear direct_reg discrete_surv gru_gap gru_d ode_rnn latent_ode latent_sde fpt_filter)
mkdir -p results/logs results/runs

die() { echo "error: $*" >&2; exit 1; }

# submit NAME "command" [dependency-job-ids...]  -> prints the job id (or runs inline with --local)
submit() {
  local name="$1" cmd="$2"; shift 2
  if [[ $LOCAL -eq 1 ]]; then echo "### $name"; bash -c "$cmd"; echo "local"; return; fi
  submit_raw "$name" "source '$RIPEN_ENV/bin/activate' && cd '$RIPEN_ROOT' && $cmd" 08:00:00 "$@"
}

# submit_raw NAME "command" TIME [dependency-job-ids...]  -> sbatch without activating the venv; prints the job id
submit_raw() {
  local name="$1" cmd="$2" time="$3"; shift 3
  [[ "${RIPEN_ACCOUNT:-CHANGE_ME}" == "CHANGE_ME" || -z "${RIPEN_ACCOUNT:-}" ]] && die "set RIPEN_ACCOUNT / RIPEN_PARTITION / RIPEN_QOS first"
  local deps=("${AFTER[@]}" "$@") dep=""; [[ ${#deps[@]} -gt 0 ]] && dep="--dependency=afterok:$(IFS=:; echo "${deps[*]}")"
  sbatch --parsable --job-name="ripen-$name" --output="results/logs/%x-%j.out" --time="$time" --cpus-per-task=4 --mem=16G \
    --gres="$RIPEN_GPU" --account="$RIPEN_ACCOUNT" --partition="$RIPEN_PARTITION" --qos="$RIPEN_QOS" $dep \
    --wrap="cd '$RIPEN_ROOT' && $cmd"
}

# setup and data are whole stages; outside --local they resubmit this script as one sbatch job
self_job() { JOBS[$1]=$(submit_raw "$1" "bash run_umiacs.sh $1 --local" "$2" "${@:3}"); }

stage_setup() {
  [[ -d "$RIPEN_ENV" ]] || "${RIPEN_PYTHON:-python3}" -m venv "$RIPEN_ENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -e ".[dev,features]"
  if command -v nvidia-smi >/dev/null 2>&1; then "$PY" -m pip install -q -U "jax[cuda12]"; fi
  JAX_PLATFORMS=cpu "$PY" -m pytest -q -m "not slow"
  JAX_PLATFORMS=cpu "$PY" scripts/smoke_all.py
}

stage_data() {
  local n_img=0; [[ -d data/avocado/images ]] && n_img=$(find data/avocado/images -name '*.jpg' | wc -l)
  [[ $n_img -ge 14710 ]] || "$PY" scripts/download_avocado.py --workers 8 || die "avocado download incomplete; rerun data"
  "$PY" -c "from ripen.data.avocado import load_avocado; n = len(load_avocado('data/avocado')); print('avocado items', n); assert n == 478"
  [[ -f data/pbc2/pbc2_cleaned.csv ]] || curl -fsSL --retry 3 --max-time 600 -o data/pbc2/pbc2_cleaned.csv --create-dirs https://raw.githubusercontent.com/chl8856/Dynamic-DeepHit/master/data/pbc2_cleaned.csv
  if [[ ! -f data/cmapss/train_FD001.txt ]]; then mkdir -p data/cmapss && curl -fsSL --retry 3 --max-time 1800 -o data/cmapss/CMAPSSData.zip https://data.nasa.gov/docs/legacy/CMAPSSData.zip && (cd data/cmapss && unzip -qo CMAPSSData.zip); fi
  [[ -d data/a2sl ]] && "$PY" scripts/ingest_a2sl.py || echo "note: data/a2sl not present; e6 will be skipped"
  # the loaders read the feature caches, so extract them before the splits
  [[ -f data/features/avocado_resnet18.npz ]] || "$PY" scripts/extract_features.py --data avocado --root data/avocado --out data/features/avocado_resnet18.npz --device "$( [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo cuda || echo cpu )"
  if [[ -d data/a2sl && ! -f data/features/a2sl_resnet18.npz ]]; then "$PY" scripts/extract_features.py --data a2sl --out data/features/a2sl_resnet18.npz --device cpu; fi
  "$PY" scripts/make_splits.py data=avocado
  for g in T10 T20 Tamb; do "$PY" scripts/make_splits.py data=avocado splits.mode=logo splits.held_out=$g; done
  "$PY" scripts/make_splits.py data=pbc2; "$PY" scripts/make_splits.py data=cmapss
}

stage_synthetic() { JOBS[synthetic]=$(submit synthetic "$PY scripts/synthetic_recovery.py"); }

stage_e1() {
  for m in "${MODELS[@]}"; do JOBS[e1_$m]=$(submit "e1-$m" "$PY scripts/train.py model=$m data=avocado"); done
  JOBS[e1_labels]=$(submit "e1-labels-only" "$PY scripts/train.py model=latent_sde data=avocado +experiment=labels_only")
}

stage_e2() {
  JOBS[e2_pde]=$(submit e2-pde "$PY scripts/train.py model=latent_sde data=avocado model.event_likelihood=pde run_name=latent_sde_avocado_pde")
  JOBS[e2_mc]=$(submit e2-mc "$PY scripts/train.py model=latent_sde data=avocado model.event_likelihood=mc run_name=latent_sde_avocado_mc")
  JOBS[e2_filter]=$(submit e2-filter "$PY scripts/train.py model=fpt_filter data=avocado run_name=fpt_filter_avocado_exact")
  JOBS[e2_synth_filter]=$(submit e2-synth-filter "$PY scripts/train.py model=fpt_filter data=synthetic run_name=fpt_filter_synthetic_exact")
  JOBS[e2_synth_pde]=$(submit e2-synth-pde "$PY scripts/train.py model=latent_sde data=synthetic model.event_likelihood=pde run_name=latent_sde_synthetic_pde")
  JOBS[e2_synth_mc]=$(submit e2-synth-mc "$PY scripts/train.py model=latent_sde data=synthetic model.event_likelihood=mc run_name=latent_sde_synthetic_mc")
}

stage_e3() {
  for g in T10 T20 Tamb; do
    for m in direct_reg gru_d latent_sde fpt_filter; do JOBS[e3_${m}_$g]=$(submit "e3-$m-$g" "$PY scripts/train.py model=$m data=avocado +experiment=logo splits.held_out=$g"); done
    for c in group fixed; do JOBS[e3_clock_${c}_$g]=$(submit "e3-clock-$c-$g" "$PY scripts/train.py model=latent_sde data=avocado +experiment=logo splits.held_out=$g model.clock=$c run_name=latent_sde_avocado_logo_${g}_clock_$c"); done
  done
}

stage_e4() {
  for k in 2 3; do for m in gru_d latent_ode latent_sde; do JOBS[e4_${m}_$k]=$(submit "e4-$m-every$k" "$PY scripts/train.py model=$m data=avocado +experiment=sampling_every$k"); done; done
  # photo-count curves come from e1's metrics_by_n_obs.json; evaluating daily-trained models on sparser visits:
  local deps=(); for m in gru_d latent_ode latent_sde; do [[ -n "${JOBS[e1_$m]:-}" && "${JOBS[e1_$m]}" != local ]] && deps+=("${JOBS[e1_$m]}"); done
  JOBS[e4_eval]=$(submit e4-eval "for m in gru_d latent_ode latent_sde; do for k in 2 3; do $PY scripts/evaluate.py results/runs/\${m}_avocado --every \$k; done; done" "${deps[@]}")
}

stage_e5() {
  for d in pbc2 cmapss; do for m in direct_reg discrete_surv gru_d ode_rnn latent_ode latent_sde fpt_filter; do
    JOBS[e5_${m}_$d]=$(submit "e5-$m-$d" "$PY scripts/train.py model=$m data=$d horizon=\$( [[ $d == pbc2 ]] && echo 180 || echo 40 ) ++model.use_stages=false")
  done; done
}

stage_e6() {
  # the manifests are committed but the raw A2SL data (scans, photos) is not; without it e6 would fail and block summary
  [[ -f data/a2sl/scans_avocado_jun2023.npz ]] || { echo "e6 skipped: data/a2sl not present (docs/data_a2sl.md)" >&2; return; }
  for m in gru_d latent_sde fpt_filter; do JOBS[e6_$m]=$(submit "e6-$m" "$PY scripts/train.py model=$m data=a2sl horizon=20 train.batch_size=8"); done
  JOBS[e6_gen]=$(submit e6-generative "$PY scripts/train.py model=latent_sde data=a2sl horizon=20 train.batch_size=8 model.decode_features=true run_name=latent_sde_a2sl_generative")
}

stage_summary() {
  local deps=(); for j in "${JOBS[@]}"; do [[ "$j" != local ]] && deps+=("$j"); done
  JOBS[summary]=$(submit summary "$PY scripts/summarize.py" "${deps[@]}")
}

case "$STAGE" in
  setup) if [[ $LOCAL -eq 1 ]]; then stage_setup; else self_job setup 02:00:00; fi ;;
  data) if [[ $LOCAL -eq 1 ]]; then stage_data; else self_job data 04:00:00; fi ;;
  pipeline)
    [[ $LOCAL -eq 1 ]] && die "pipeline is the sbatch chain; use setup/data/all --local instead"
    self_job setup 02:00:00; self_job data 04:00:00 "${JOBS[setup]}"; AFTER+=("${JOBS[data]}")
    stage_synthetic; stage_e1; stage_e2; stage_e3; stage_e4; stage_e5; stage_e6; stage_summary ;;
  synthetic) stage_synthetic ;;
  e1) stage_e1 ;;
  e2) stage_e2 ;;
  e3) stage_e3 ;;
  e4) stage_e4 ;;
  e5) stage_e5 ;;
  e6) stage_e6 ;;
  summary) stage_summary ;;
  all) stage_synthetic; stage_e1; stage_e2; stage_e3; stage_e4; stage_e5; stage_e6; stage_summary ;;
  *) die "unknown stage $STAGE" ;;
esac
[[ $LOCAL -eq 1 ]] || { echo "submitted:"; for k in "${!JOBS[@]}"; do echo "  $k -> ${JOBS[$k]}"; done | sort; }
