#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH -J "benbio"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

# module load python3/3.7.0
# module load gcc/9.2.0

round_dir="/groups/astuart/mlevine/dimer_computations_colab/results/metaclusters"

echo "Sending results to $round_dir"
srun --ntasks=1 python3 metacluster_inference_tests_1.py --n_targets_per_round 100 --maxiter_O 5 --popsize_O 10 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/inner5x10_outer10x25_100targets"
wait
