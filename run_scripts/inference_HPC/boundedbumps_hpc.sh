#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --array=0-1763      # how many tasks in the array
#SBATCH -J "boundebumps"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

base_dir="/groups/astuart/mlevine/dimer_computations_colab/results/BoundedBumps_0.2"

srun python bump_inference_tests_ainner_BoundedBumpsSeparate_main.py --id $SLURM_ARRAY_TASK_ID --base_dir $base_dir
