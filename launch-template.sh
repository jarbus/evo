#!/bin/bash
# shellcheck disable=SC2206

#SBATCH --mail-type=END
#SBATCH --mail-user=9147037394@vtext.com
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=/home/garbus/evotrade/runs/${JOB_NAME}.std
#SBATCH --account=guest
#SBATCH --time=24:00:00
#SBATCH --partition=guest-compute
#SBATCH --ntasks=${NPROCS}
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3GB

source /home/garbus/.bashrc
conda activate trade
julia run-script.jl $(cat /home/garbus/evotrade/afiles/${RUN_NAME}.arg) --exp-name "${RUN_NAME}" --datime "${DATIME}"
