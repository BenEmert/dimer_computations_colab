#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
#SBATCH --array=0-8
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

base_dir="/groups/astuart/mlevine/dimer_computations_colab/results/bigSims"

srun python script.py --n_monomers 9 --base_dir $base_dir --job_id $SLURM_ARRAY_TASK_ID
