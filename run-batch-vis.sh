#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --output=/home/garbus/evotrade/run-batch-vis.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

 for out in $1/*.out ; do
    python /home/garbus/evotrade/s2g.py "$out" &
 done

wait
