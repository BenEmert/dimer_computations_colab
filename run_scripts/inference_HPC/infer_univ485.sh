#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "benbio"   # job name
#SBATCH --output=slurm/out-%x.%j.out
#SBATCH --error=slurm/err-%x.%j.err

module load python3/3.7.0
module load gcc/9.2.0

round_dir="/groups/astuart/mlevine/benbio/results/true_inference"

echo "Sending results to $round_dir"
srun --ntasks=1 python3 inference_tests_3m_univ485.py --maxiter_K 15 --popsize_K 50 --base_dir "$round_dir/univ485_15x50_15x50"
wait
