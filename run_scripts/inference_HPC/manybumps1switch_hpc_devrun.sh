#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=10:00:00   # walltime
#SBATCH --array=0-71      # how many tasks in the array
#SBATCH -J "manybumps1switch"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

base_dir="/groups/astuart/mlevine/dimer_computations_colab/results/ManyBumps1switch_3.1_devrun"

srun python bump_inference_tests_ainner_ManyBumps_main.py --id $SLURM_ARRAY_TASK_ID --base_dir $base_dir --dev_run 1