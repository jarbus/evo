#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --output=/dev/null

 for out in $1/*.out ; do
    python /home/garbus/evotrade/s2g.py "$out"
 done

