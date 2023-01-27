#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p shared,tambe,seas_compute,serial_requeue
#SBATCH -t 5:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=200G          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o Phase3/Bash/Outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e Phase3/Bash/Errors/%A-%a.err  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
set -x
date

source activate env2
echo ${1}
python3 Phase3/Code/Run.py ${1} ${SLURM_ARRAY_TASK_ID}