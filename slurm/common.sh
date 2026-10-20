#!/usr/bin/env bash
# Shared SLURM preamble for UMIACS. Edit the CHANGE_ME values once; every job sources this file.
#
#   ACCOUNT / PARTITION / QOS : from `sacctmgr show assoc user=$USER` on the cluster
#   RIPEN_ENV                 : path of the Python env with `pip install -e ".[dev,features]"` done

export RIPEN_ACCOUNT="${RIPEN_ACCOUNT:-CHANGE_ME}"
export RIPEN_PARTITION="${RIPEN_PARTITION:-CHANGE_ME}"
export RIPEN_QOS="${RIPEN_QOS:-CHANGE_ME}"
export RIPEN_ENV="${RIPEN_ENV:-$HOME/envs/ripen}"
export RIPEN_ROOT="${RIPEN_ROOT:-$HOME/ripen}"

activate() {
  cd "$RIPEN_ROOT" || exit 1
  # shellcheck disable=SC1091
  source "$RIPEN_ENV/bin/activate"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PYTHONUNBUFFERED=1
  mkdir -p results/logs
}
