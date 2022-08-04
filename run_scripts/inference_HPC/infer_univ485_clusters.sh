#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH -J "benbio"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

# module load python3/3.7.0
# module load gcc/9.2.0

round_dir="/groups/astuart/mlevine/dimer_computations_colab/results/true_inference"

echo "Sending results to $round_dir"
srun --ntasks=1 python3 inference_tests_3m_univ485_cluster.py --n_weights_per_round 10 --maxiter_K 15 --popsize_K 50 --base_dir "$round_dir/univ485_clusters_15x50_15x50_10targets"
wait
