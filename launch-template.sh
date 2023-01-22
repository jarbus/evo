#!/bin/bash
# shellcheck disable=SC2206

#SBATCH --mail-type=END
#SBATCH --mail-user=9147037394@vtext.com
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=/home/garbus/evotrade/runs/${CLS_NAME}/${JOB_NAME}.std
#SBATCH --error=/home/garbus/evotrade/runs/${CLS_NAME}/${JOB_NAME}.err
#SBATCH --account=guest
#SBATCH --time=24:00:00
#SBATCH --partition=guest-compute
#SBATCH --ntasks=${NPROCS}
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6GB
#SBATCH --exclude=compute-5-[0,12],compute-7-0

source /home/garbus/.bashrc
conda activate trade
julia run-script.jl $(cat /home/garbus/evotrade/afiles/${CLS_NAME}/${RUN_NAME}.arg | grep "^[^#]") --cls-name "${CLS_NAME}" --exp-name "${RUN_NAME}" --datime "${DATIME}"
