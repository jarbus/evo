#!/bin/bash
# shellcheck disable=SC2206

#SBATCH --mail-type=END
#SBATCH --mail-user=9147037394@vtext.com
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=/home/garbus/evotrade/all.log
#SBATCH --account=guest
#SBATCH --time=24:00:00
#SBATCH --partition=guest-compute
#SBATCH --ntasks=40                                          │
#SBATCH --cpus-per-task=5 
#SBATCH --mem-per-cpu=3GB
#SBATCH --exclude=gpu-6-9,compute-9-[0-5]                    │

source /home/garbus/.bashrc
conda activate trade
julia run-script.jl $(cat /home/garbus/evotrade/afiles/${JOB_NAME}.arg) --exp-name ${JOB_NAME}
